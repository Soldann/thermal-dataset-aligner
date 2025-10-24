import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFile
# Load configuration
import configparser
import ast
config = configparser.ConfigParser()
config.read("./config.ini")

# Handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set deterministic behavior for reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(mode=True, warn_only=True)


GROUND_IMAGE_SIZE = ast.literal_eval(config.get("VIGOR", "GROUND_IMAGE_SIZE"))
SATELLITE_IMAGE_SIZE = ast.literal_eval(config.get("VIGOR", "SATELLITE_IMAGE_SIZE"))


# Define transformations
transform_grd = transforms.Compose([
    transforms.Resize(GROUND_IMAGE_SIZE),
    transforms.ToTensor()
])

transform_sat = transforms.Compose([
    transforms.Resize(SATELLITE_IMAGE_SIZE),
    transforms.ToTensor()
])


class VIGORDataset(Dataset):
    def __init__(self, root='/home/ziminxia/Work/datasets/VIGOR', label_root='splits_new', 
                 split='samearea', train=True, random_orientation=0, transform=None, pos_only=True):
        self.root = root
        self.label_root = label_root
        self.split = split
        self.train = train
        self.pos_only = pos_only
        self.random_orientation = random_orientation
        self.grdimage_transform, self.satimage_transform = transform if transform else (transform_grd, transform_sat)

        self.city_list = self._get_city_list()
        self.sat_list, self.sat_index_dict = self._load_satellite_data()
        self.grd_list, self.label, self.delta, self.sat_cover_dict = self._load_ground_data()
        self.data_size = len(self.grd_list)

    def _get_city_list(self):
        if self.split == 'samearea':
            # return ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
            return ['NewYork']
        if self.split == 'crossarea':
            return ['NewYork', 'Seattle'] if self.train else ['SanFrancisco', 'Chicago']
        return []

    def _load_satellite_data(self):
        sat_list, sat_index_dict = [], {}
        idx = 0
        
        for city in self.city_list:
            sat_list_fname = os.path.join(self.root, self.label_root, city, 'satellite_list.txt')
            with open(sat_list_fname, 'r') as file:
                for line in file:
                    sat_path = os.path.join(self.root, city, 'satellite', line.strip())
                    sat_list.append(sat_path)
                    sat_index_dict[line.strip()] = idx
                    idx += 1
            print(f'Loaded {sat_list_fname}, {idx} entries')
        
        return np.array(sat_list), sat_index_dict

    def _load_ground_data(self):
        grd_list, label_list, delta_list = [], [], []
        sat_cover_dict = {}
        idx = 0

        for city in self.city_list:
            label_fname = self._get_label_file(city)
            
            with open(label_fname, 'r') as file:
                for line in file:
                    data = np.array(line.split())
                    label = np.array([self.sat_index_dict[data[i]] for i in [1, 4, 7, 10]]).astype(int)
                    delta = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(np.float32)
                    
                    grd_list.append(os.path.join(self.root, city, 'panorama', data[0]))
                    label_list.append(label)
                    delta_list.append(delta)
                    
                    if label[0] not in sat_cover_dict:
                        sat_cover_dict[label[0]] = [idx]
                    else:
                        sat_cover_dict[label[0]].append(idx)
                    idx += 1
            
            print(f'Loaded {label_fname}, {idx} entries')
        
        return grd_list, np.array(label_list), np.array(delta_list), sat_cover_dict

    def _get_label_file(self, city):
        if self.split == 'samearea':
            return os.path.join(self.root, self.label_root, city, 'same_area_balanced_train.txt' if self.train else 'same_area_balanced_test.txt')
        return os.path.join(self.root, self.label_root, city, 'pano_label_balanced.txt')

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        idx = 0
        grd = self._load_image(self.grd_list[idx], default_size=GROUND_IMAGE_SIZE)
        grd = self.grdimage_transform(grd)
        rotation = np.random.uniform(-self.random_orientation/360, self.random_orientation/360)
        grd_rolled = torch.roll(grd, int(round(rotation * grd.size(2))), dims=2)
        yaw = -rotation * 360 * (np.pi/180)  # 0 means heading North, clockwise increasing

        sat, row_offset, col_offset = self._load_satellite_image(idx)

        gt_loc = torch.tensor([[-row_offset, col_offset]])
        r = torch.tensor([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]).to(torch.float32)
        
        plt.imshow(grd_rolled.permute(1, 2, 0))
        city = self._get_city_name(self.grd_list[idx])
        
        return grd_rolled, grd, gt_loc, r, city

    def _load_image(self, path, default_size):
        try:
            img = Image.open(path).convert('RGB')
        except:
            print(f'Unreadable image: {path}')
            img = Image.new('RGB', default_size)
        return img

    def _load_satellite_image(self, idx):
        if self.pos_only:
            pos_index = 0
            row_offset, col_offset = self.delta[idx, pos_index]
        else:
            raise NotImplementedError("This function has not been implemented yet.")
        # else:
        #     pos_index, row_offset, col_offset = self._get_valid_satellite_index(idx)

        sat = self._load_image(self.sat_list[self.label[idx][pos_index]], default_size=SATELLITE_IMAGE_SIZE)
        width_raw, height_raw = sat.size[::-1]
        
        sat = self.satimage_transform(sat)

        row_offset = row_offset / height_raw * sat.size(1)
        col_offset = col_offset / width_raw * sat.size(2)
        
        return sat, row_offset, col_offset

    # def _get_valid_satellite_index(self, idx):
    #     while True:
    #         pos_index = random.randint(0, 3)
    #         row_offset, col_offset = self.delta[idx, pos_index]
    #         if abs(col_offset) < 320 and abs(row_offset) < 320:
    #             return pos_index, row_offset, col_offset

    def _get_city_name(self, path):
        for city in ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']:
            if city in path:
                return city
        return 'Unknown'
