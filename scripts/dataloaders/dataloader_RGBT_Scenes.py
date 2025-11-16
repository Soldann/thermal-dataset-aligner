import os
import random
import numpy as np
import csv
import torch
import json
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
import cv2 
from pathlib import Path
import math
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
    def __init__(self, root, split='train', transform=None, low_memory_mode=False):
        self.root: Path = Path(root)
        self.split = split
        self.data_correspondencer = None

        # self.thermal_images = glob.glob(os.path.join(self.root, 'thermal', 'train', '*.jpg')) + glob.glob(os.path.join(self.root, 'thermal', 'test', '*.jpg'))
        self.thermal_images = glob.glob(str(self.root / 'thermal' / 'test' / '*.jpg'))
        self.rgb_images = glob.glob(str(self.root / 'rgb' / 'test' / '*.jpg'))
        # self.rgb_images = glob.glob(os.path.join(self.root, 'rgb', 'train', '*.jpg')) + glob.glob(os.path.join(self.root, 'rgb', 'test', '*.jpg'))
        self.thermal_images.sort()
        self.rgb_images.sort()

        # Sample from all combinations of thermal images within a sliding window of 5 images
        self.image_pairs = []
        window_size = 5
        for i in range(len(self.thermal_images)):
            for j in range(i+1, min(i + window_size, len(self.thermal_images))):
                self.image_pairs.append((i, j))

        # self.image_pairs = self.image_pairs[:10] # For testing, limit to first 10 pairs
        self.image1_list = []
        self.image2_list = []
        self.kpts1_list = []
        self.kpts2_list = []
        self.conf_list = []
        self.patches1_list = []
        self.patches2_list = []
        (self.root / self.split / 'keypoint_cache').mkdir(parents=True, exist_ok=True)
        if low_memory_mode:
            for image1, image2 in self.image_pairs:
                print("processing image pair:", image1, image2, "corresponding to files:", self.thermal_images[image1], self.thermal_images[image2])
                if not (self.root / self.split / 'keypoint_cache' / f"{image1}_{image2}_keypoints.pt").exists():
                    if self.data_correspondencer is None:
                        self.data_correspondencer = DatasetCorrespondencer(VGGTFeatureMatcher(), CombinedAligner())
                    img1_list, img2_list, kpts1_list, kpts2_list, conf_list = self.data_correspondencer.compute_correspondences([self.rgb_images[image1], self.rgb_images[image2]], [self.thermal_images[image1], self.thermal_images[image2]], [ImageModality.thermal]*2)
                    patches_1, patches_2 = [], []
                    for img1, kpts1, kpts2 in zip(img1_list, kpts1_list, kpts2_list):
                        p1, p2 = compute_patch_matches(kpts1, kpts2, img1.shape[1:], patch_size=14) # img1 shape is (C, H, W)
                        patches_1.append(p1)
                        patches_2.append(p2)
                    
                    torch.save((img1_list, img2_list, kpts1_list, kpts2_list, conf_list, patches_1, patches_2), self.root / self.split / 'keypoint_cache' / f"{image1}_{image2}_keypoints.pt")
                else:
                    print("loading cached keypoints for image pair:", image1, image2)
                    img1_list, img2_list, kpts1_list, kpts2_list, conf_list, patches_1, patches_2 = torch.load(self.root / self.split / 'keypoint_cache' / f"{image1}_{image2}_keypoints.pt")
                
                if kpts1_list[0].shape[0] == 0 or kpts2_list[0].shape[0] == 0:
                    print(f"No keypoints found for image pair: {image1}, {image2}. Skipping.")
                    continue
                    # raise ValueError("No keypoints found.")
                
                self.image1_list.extend([img1.cpu() for img1 in img1_list]) # Make sure on cpu
                self.image2_list.extend([img2.cpu() for img2 in img2_list])
                self.kpts1_list.extend([kpts1.cpu() for kpts1 in kpts1_list])
                self.kpts2_list.extend([kpts2.cpu() for kpts2 in kpts2_list])
                self.conf_list.extend([conf.cpu() for conf in conf_list])
                self.patches1_list.extend([p1.cpu() for p1 in patches_1])
                self.patches2_list.extend([p2.cpu() for p2 in patches_2])
        else:
            img1_list, img2_list, kpts1_list, kpts2_list, conf_list = self.data_correspondencer.compute_correspondences(self.rgb_images, self.thermal_images, [ImageModality.thermal]*len(self.thermal_images), self.image_pairs)
            self.image1_list = [img1.cpu() for img1 in img1_list]
            self.image2_list = [img2.cpu() for img2 in img2_list]
            self.kpts1_list = [kpts1.cpu() for kpts1 in kpts1_list]
            self.kpts2_list = [kpts2.cpu() for kpts2 in kpts2_list]
            self.conf_list = [conf.cpu() for conf in conf_list]
            patches_1, patches_2 = [], []
            for img1, kpts1, kpts2 in zip(img1_list, kpts1_list, kpts2_list):
                p1, p2 = compute_patch_matches(kpts1, kpts2, img1.shape[2:], patch_size=14)
                patches_1.append(p1)
                patches_2.append(p2)
            self.patches1_list.extend([p1.cpu() for p1 in patches_1])
            self.patches2_list.extend([p2.cpu() for p2 in patches_2])

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
        return len(self.image1_list)

    def __getitem__(self, idx):
        return self.image1_list[idx], self.image2_list[idx], self.conf_list[idx], self.patches1_list[idx], self.patches2_list[idx], self.patches1_list[idx].shape[0], self.patches2_list[idx].shape[0]

if __name__ == "__main__":
    data = RGBT_Scenes_Dataset(
        root='/home/landson/RGBT-Scenes/Building',
        split='train',
        low_memory_mode=True
    )
    print(len(data))