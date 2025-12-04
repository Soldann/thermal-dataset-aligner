import os
import random
import numpy as np
import csv
import torch
import json
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
import cv2 
from pathlib import Path
import math
from enum import Enum
import re
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys
import os

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(parent_dir)

# Import our custom modules
from dataset_correspondencer import DatasetCorrespondencer, ImageModality
from alignment_methods.combined_aligner import CombinedAligner
from alignment_methods.dataset_aligner_base import DatasetAligner
from feature_matchers.base_feature_matcher import FeatureMatcher
from feature_matchers.vggt_feature_matcher import VGGTFeatureMatcher
import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from keypoint_patches import compute_patch_matches
import glob

# Set deterministic behavior for reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(mode=True, warn_only=True)

class RGBT_Scenes_Dataset(Dataset):
    class ImagePairMode(Enum):
        thermal_to_thermal = "thermal_to_thermal"
        rgb_to_thermal = "rgb_to_thermal"

    @staticmethod
    def build_test_train_dataloaders(root, training_ratio=0.7, low_memory_mode=False):
        root: Path = Path(root)

        thermal_images = glob.glob(os.path.join(root, 'thermal', 'train', '*.jpg')) + glob.glob(os.path.join(root, 'thermal', 'test', '*.jpg'))
        rgb_images = glob.glob(os.path.join(root, 'rgb', 'train', '*.jpg')) + glob.glob(os.path.join(root, 'rgb', 'test', '*.jpg'))

        # pick a random subset of training_ratio of the data for training
        randomized_indices = torch.randperm(len(thermal_images))
        train_size = int(training_ratio * len(thermal_images))
        train_indices = randomized_indices[:train_size]
        val_indices = randomized_indices[train_size:]
        train_dataset = RGBT_Scenes_Dataset(root, [Path(thermal_images[i]) for i in train_indices], [Path(rgb_images[i]) for i in train_indices], low_memory_mode=low_memory_mode)
        val_dataset = RGBT_Scenes_Dataset(root, [Path(thermal_images[i]) for i in val_indices], [Path(rgb_images[i]) for i in val_indices], low_memory_mode=low_memory_mode)
        return train_dataset, val_dataset

    def __init__(self, root, thermal_images, rgb_images, window_size=10, low_memory_mode=False, image_mode=ImagePairMode.thermal_to_thermal):
        self.keypoint_cache_path: Path = Path(root) / 'keypoint_cache' / image_mode.value
        self.data_correspondencer = None

        self.thermal_images = thermal_images
        self.rgb_images = rgb_images
        self.thermal_images.sort()
        self.rgb_images.sort()

        # Sample from all combinations of thermal images within a sliding window of 5 images
        self.image_pairs = []
        self.image_pair_indexes = []
        digit_pattern = r'\d+'
        for i in range(len(self.thermal_images)):
            dataset_image1_num = int(re.search(digit_pattern, self.thermal_images[i].stem).group())
            for j in range(i+1, min(i + window_size, len(self.thermal_images))):
                dataset_image2_num = int(re.search(digit_pattern, self.thermal_images[j].stem).group())
                if abs(dataset_image1_num - dataset_image2_num) < window_size:
                    self.image_pairs.append((self.thermal_images[i].stem, self.thermal_images[j].stem))
                    self.image_pair_indexes.append((i, j))

        # self.image_pairs = self.image_pairs[:10] # For testing, limit to first 10 pairs
        (self.keypoint_cache_path).mkdir(parents=True, exist_ok=True)
        if low_memory_mode:
            validated_pairs = []
            validated_pair_indexes = []
            for (image1, image2), (img1_idx, img2_idx) in zip(self.image_pairs, self.image_pair_indexes):
                print("processing image pair:", image1, image2)
                if not (self.keypoint_cache_path / f"{image1}_{image2}_keypoints.pt").exists():
                    if self.data_correspondencer is None:
                        self.data_correspondencer = DatasetCorrespondencer(VGGTFeatureMatcher(), CombinedAligner())
                    
                    if image_mode == RGBT_Scenes_Dataset.ImagePairMode.thermal_to_thermal:
                        img1_list, img2_list, kpts1_list, kpts2_list, conf_list = self.data_correspondencer.compute_correspondences([self.rgb_images[img1_idx], self.rgb_images[img2_idx]], [self.thermal_images[img1_idx], self.thermal_images[img2_idx]], [ImageModality.thermal]*2)
                    elif image_mode == RGBT_Scenes_Dataset.ImagePairMode.rgb_to_thermal:
                        # randomly pick one rgb and one thermal image
                        modalities = [ImageModality.rgb, ImageModality.thermal]
                        randomized_modalities = modalities[torch.randperm(len(modalities))]
                        img1_list, img2_list, kpts1_list, kpts2_list, conf_list =  self.data_correspondencer.compute_correspondences([self.rgb_images[img1_idx], self.rgb_images[img2_idx]], [self.thermal_images[img1_idx], self.thermal_images[img2_idx]], randomized_modalities)
                    else:
                        raise ValueError("Invalid image mode.")

                    patches_1, patches_2 = [], []
                    for img1, kpts1, kpts2 in zip(img1_list, kpts1_list, kpts2_list):
                        p1, p2 = compute_patch_matches(kpts1, kpts2, img1.shape[1:], patch_size=14) # img1 shape is (C, H, W)
                        patches_1.append(p1)
                        patches_2.append(p2)
                    
                    torch.save((img1_list, img2_list, kpts1_list, kpts2_list, conf_list, patches_1, patches_2), self.keypoint_cache_path / f"{image1}_{image2}_keypoints.pt")
                else:
                    print("loading cached keypoints for image pair:", image1, image2)
                    img1_list, img2_list, kpts1_list, kpts2_list, conf_list, patches_1, patches_2 = torch.load(self.keypoint_cache_path / f"{image1}_{image2}_keypoints.pt")
                
                if kpts1_list[0].shape[0] == 0 or kpts2_list[0].shape[0] == 0:
                    print(f"No keypoints found for image pair: {image1}, {image2}. Skipping.")
                    continue
                else:
                    validated_pairs.append((image1, image2))
                    validated_pair_indexes.append((img1_idx, img2_idx))
                    # raise ValueError("No keypoints found.")
            self.image_pairs = validated_pairs
            self.image_pair_indexes = validated_pair_indexes
        else:
            if image_mode == RGBT_Scenes_Dataset.ImagePairMode.thermal_to_thermal:
                img1_list, img2_list, kpts1_list, kpts2_list, conf_list = self.data_correspondencer.compute_correspondences(self.rgb_images, self.thermal_images, [ImageModality.thermal]*len(self.thermal_images), self.image_pair_indexes)
            elif image_mode == RGBT_Scenes_Dataset.ImagePairMode.rgb_to_thermal:
                randomized_modalities_list = torch.randint(0, ImageModality.num_modalities.value, (len(self.thermal_images),))
                img1_list, img2_list, kpts1_list, kpts2_list, conf_list = self.data_correspondencer.compute_correspondences(self.rgb_images, self.thermal_images, randomized_modalities_list, self.image_pair_indexes)

            patches_1, patches_2 = [], []
            for img1, kpts1, kpts2 in zip(img1_list, kpts1_list, kpts2_list):
                p1, p2 = compute_patch_matches(kpts1, kpts2, img1.shape[2:], patch_size=14)
                patches_1.append(p1)
                patches_2.append(p2)

            for i, (image1, image2) in enumerate(self.image_pairs):
                print("saving precomputed keypoints for image pair:", image1, image2)
                torch.save((img1_list[i], img2_list[i], kpts1_list[i], kpts2_list[i], conf_list[i], patches_1[i], patches_2[i]), self.keypoint_cache_path / f"{image1}_{image2}_keypoints.pt")

    def collate_fn(self, batch):
        images1, images2, conf, patches1, patches2, patches1_length, patches2_length = zip(*batch)

        # Stack images into batch tensors
        images1 = torch.stack(images1, dim=0)
        images2 = torch.stack(images2, dim=0)
        conf = torch.stack(conf, dim=0)

        padded_patches1 = pad_sequence(patches1, batch_first=True)
        padded_patches2 = pad_sequence(patches2, batch_first=True)

        patches1_length = torch.tensor(patches1_length)
        patches2_length = torch.tensor(patches2_length)

        return images1, images2, conf, padded_patches1, padded_patches2, patches1_length, patches2_length

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        image1, image2 = self.image_pairs[idx]
        img1_list, img2_list, kpts1_list, kpts2_list, conf_list, patches_1, patches_2 = torch.load(self.keypoint_cache_path / f"{image1}_{image2}_keypoints.pt", map_location='cpu')
        # print("loaded data for image pair:", image1, image2)
        return img1_list[0], img2_list[0], conf_list[0], patches_1[0], patches_2[0], patches_1[0].shape[0], patches_2[0].shape[0]

if __name__ == "__main__":
    train_data, val_data = RGBT_Scenes_Dataset.build_test_train_dataloaders("/home/landson/RGBT-Scenes/Building", training_ratio=0.7, low_memory_mode=True)
    print(len(train_data), 'training samples found.')
    print(len(val_data), 'validation samples found.')