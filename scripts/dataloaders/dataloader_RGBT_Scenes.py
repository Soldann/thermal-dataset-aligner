import os
import random
import numpy as np
import csv
import torch
import json
from torch.utils.data import Dataset, DataLoader
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
from alignment_methods.feature_aligner import FeatureAligner
from alignment_methods.dataset_aligner_base import DatasetAligner
from feature_matchers.base_feature_matcher import FeatureMatcher
from feature_matchers.vggt_feature_matcher import VGGTFeatureMatcher

import glob

# Load configuration
import configparser
import ast
config = configparser.ConfigParser()
config.read("./config.ini")


# Set deterministic behavior for reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(mode=True, warn_only=True)

# ground_image_size = ast.literal_eval(config.get("RobotCar", "ground_image_size"))
# satellite_image_size = ast.literal_eval(config.get("RobotCar", "satellite_image_size"))

# Define transformations
# transform_grd = transforms.Compose([
#     transforms.Resize(ground_image_size),
#     transforms.ToTensor()
# ])

# transform_sat = transforms.Compose([
#     transforms.Resize(satellite_image_size),
#     transforms.ToTensor()
# ])

class RGBT_Scenes_Dataset(Dataset):
    def __init__(self, root, split='train', transform=None, low_memory_mode=False):
        self.root: Path = Path(root)
        self.split = split
        # self.grdimage_transform, self.satimage_transform = transform if transform else (None, None)

        # self.grdList = self._load_ground_list()
        # self.grdYaw = self._load_yaw()

        # self.grdNum = len(self.grdList)
        # grdarray = np.array(self.grdList)
        # self.grdUTM = np.transpose(grdarray[:, 2:].astype(np.float64))

        # self.transform_coords = self._compute_coord_transform()

        # self.meters_per_pixel = 0.09240351462361521 * 800 / satellite_image_size[0]
        self.data_correspondencer = DatasetCorrespondencer(VGGTFeatureMatcher(), FeatureAligner())

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

        self.image1_list = []
        self.image2_list = []
        self.kpts1_list = []
        self.kpts2_list = []
        self.conf_list = []
        (self.root / self.split / 'keypoint_cache').mkdir(parents=True, exist_ok=True)
        if low_memory_mode:
            for image1, image2 in self.image_pairs:
                print("processing image pair:", image1, image2, "corresponding to files:", self.thermal_images[image1], self.thermal_images[image2])
                if not (self.root / self.split / 'keypoint_cache' / f"{image1}_{image2}_keypoints.pt").exists():
                    img1_list, img2_list, kpts1_list, kpts2_list, conf_list = self.data_correspondencer.compute_correspondences([self.rgb_images[image1], self.rgb_images[image2]], [self.thermal_images[image1], self.thermal_images[image2]], [ImageModality.thermal]*2)
                    torch.save((img1_list, img2_list, kpts1_list, kpts2_list, conf_list), self.root / self.split / 'keypoint_cache' / f"{image1}_{image2}_keypoints.pt")
                else:
                    print("loading cached keypoints for image pair:", image1, image2)
                    img1_list, img2_list, kpts1_list, kpts2_list, conf_list = torch.load(self.root / self.split / 'keypoint_cache' / f"{image1}_{image2}_keypoints.pt")
                
                if kpts1_list[0].shape[0] == 0 or kpts2_list[0].shape[0] == 0:
                    print(f"No keypoints found for image pair: {image1}, {image2}. Skipping.")
                    raise ValueError("No keypoints found.")
                
                self.image1_list.extend([img1.cpu() for img1 in img1_list]) # Make sure on cpu
                self.image2_list.extend([img2.cpu() for img2 in img2_list])
                self.kpts1_list.extend([kpts1.cpu() for kpts1 in kpts1_list])
                self.kpts2_list.extend([kpts2.cpu() for kpts2 in kpts2_list])
                self.conf_list.extend([conf.cpu() for conf in conf_list])
        else:
            img1_list, img2_list, kpts1_list, kpts2_list, conf_list = self.data_correspondencer.compute_correspondences(self.rgb_images, self.thermal_images, [ImageModality.thermal]*len(self.thermal_images), self.image_pairs)
            self.image1_list = [img1.cpu() for img1 in img1_list]
            self.image2_list = [img2.cpu() for img2 in img2_list]
            self.kpts1_list = [kpts1.cpu() for kpts1 in kpts1_list]
            self.kpts2_list = [kpts2.cpu() for kpts2 in kpts2_list]
            self.conf_list = [conf.cpu() for conf in conf_list]

    def get_full_satellite_map(self):
        if not hasattr(self, '_full_satellite_map'):
            # print(f"[{os.getpid()}] Loading satellite map into memory...")
            self._full_satellite_map = Image.open(os.path.join(self.root, 'satellite_map_new.png'))
        return self._full_satellite_map
        
    def _load_ground_list(self):
        def read_list(file_path):
            with open(file_path, 'r') as f:
                return [line.strip().split() for line in f.readlines()]

        if self.split == 'train':
            return read_list(os.path.join(self.root, 'training.txt'))
        elif self.split == 'val':
            return read_list(os.path.join(self.root, 'validation.txt'))
        elif self.split == 'test':
            test1 = read_list(os.path.join(self.root, 'test1_j.txt'))
            test2 = read_list(os.path.join(self.root, 'test2_j.txt'))
            test3 = read_list(os.path.join(self.root, 'test3_j.txt'))
            self.test1_len, self.test2_len, self.test3_len = len(test1), len(test2), len(test3)
            return test1 + test2 + test3
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def _load_yaw(self):
        yaw_file = {
            'train': 'train_yaw.npy',
            'val': 'val_yaw.npy',
            'test': 'test_yaw.npy'
        }[self.split]
        return np.load(os.path.join(self.root, yaw_file))

    def _compute_coord_transform(self):
        primary = np.array([
            [619400., 5736195.],
            [619400., 5734600.],
            [620795., 5736195.],
            [620795., 5734600.],
            [620100., 5735400.]
        ])
        secondary = np.array([
            [900., 900.], [492., 18168.], [15966., 1260.],
            [15553., 18528.], [8255., 9688.]
        ])

        pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
        X, Y = pad(primary), pad(secondary)
        A, *_ = np.linalg.lstsq(X, Y, rcond=None)
        return lambda x: np.dot(pad(x), A)[:, :-1]

    def __len__(self):
        return len(self.image1_list)

    def __getitem__(self, idx):
        return self.image1_list[idx], self.image2_list[idx], self.kpts1_list[idx], self.kpts2_list[idx], self.conf_list[idx]

    def _get_train_crop(self, image_coord):
        alpha = 2 * math.pi * random.random()
        r = 200 * math.sqrt(2) * random.random()
        row_offset, col_offset = int(r * math.cos(alpha)), int(r * math.sin(alpha))
        row, col = int(image_coord[1] + row_offset), int(image_coord[0] + col_offset)
        full_map = self.get_full_satellite_map()
        crop = full_map.crop((col - 400, row - 400, col + 400, row + 400))

        row_offset_resized = (400 + row_offset) / 800 * satellite_image_size[0] - satellite_image_size[0]/2
        col_offset_resized = (400 + col_offset) / 800 * satellite_image_size[0] - satellite_image_size[0]/2
        return crop, row_offset_resized, col_offset_resized

    def _get_val_test_crop(self, image_coord):
        col_split = int(image_coord[0] // 400)
        if image_coord[0] - 400 * col_split < 200:
            col_split -= 1
        row_split = int(image_coord[1] // 400)
        if image_coord[1] - 400 * row_split < 200:
            row_split -= 1

        col_pixel = int(np.round(image_coord[0] - 400 * col_split))
        row_pixel = int(np.round(image_coord[1] - 400 * row_split))
        full_map = self.get_full_satellite_map()
        crop = full_map.crop((
            col_split * 400, row_split * 400,
            col_split * 400 + 800, row_split * 400 + 800
        ))

        row_offset_resized = int(-(row_pixel / 800 * satellite_image_size[0] - satellite_image_size[0]/2))
        col_offset_resized = int(-(col_pixel / 800 * satellite_image_size[0] - satellite_image_size[0]/2))
        return crop, row_offset_resized, col_offset_resized

    def _get_orientation_angle(self, yaw_rad):
        angle_deg = (yaw_rad / np.pi * 180) - 90 # 0 means heading north, clockwise increasing, degrees
        return angle_deg * (np.pi/180)

if __name__ == "__main__":
    data = RGBT_Scenes_Dataset(
        root='/home/landson/RGBT-Scenes/Building',
        split='train',
        low_memory_mode=True
    )
    print(len(data))