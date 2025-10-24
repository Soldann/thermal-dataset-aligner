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

ground_image_size = ast.literal_eval(config.get("RobotCar", "ground_image_size"))
satellite_image_size = ast.literal_eval(config.get("RobotCar", "satellite_image_size"))

# Define transformations
transform_grd = transforms.Compose([
    transforms.Resize(ground_image_size),
    transforms.ToTensor()
])

transform_sat = transforms.Compose([
    transforms.Resize(satellite_image_size),
    transforms.ToTensor()
])

class RobotCarDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.grdimage_transform, self.satimage_transform = transform if transform else (None, None)

        self.grdList = self._load_ground_list()
        self.grdYaw = self._load_yaw()

        self.grdNum = len(self.grdList)
        grdarray = np.array(self.grdList)
        self.grdUTM = np.transpose(grdarray[:, 2:].astype(np.float64))

        self.transform_coords = self._compute_coord_transform()

        self.meters_per_pixel = 0.09240351462361521 * 800 / satellite_image_size[0]

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
        return self.grdNum

    def __getitem__(self, idx):
        grd_path = os.path.join(self.root, self.grdList[idx][0]) 
        grd_img = Image.open(grd_path).convert('RGB')
        grd_img = self.grdimage_transform(grd_img)

        easting, northing = self.grdUTM[:, idx]
        image_coord = self.transform_coords(np.array([[easting, northing]]))[0]

        if self.split == 'train':
            sat_crop, row_offset_resized, col_offset_resized = self._get_train_crop(image_coord)
        else:
            sat_crop, row_offset_resized, col_offset_resized = self._get_val_test_crop(image_coord)

        sat_img = self.satimage_transform(sat_crop)
        _, width, height = sat_img.size()

        gt_loc = torch.tensor([[row_offset_resized, col_offset_resized]])
        ori_angle = self._get_orientation_angle(self.grdYaw[idx])
        r = torch.tensor([[np.cos(ori_angle), -np.sin(ori_angle)], [np.sin(ori_angle), np.cos(ori_angle)]]).to(torch.float32)

        return grd_img, sat_img, gt_loc, r

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

    
