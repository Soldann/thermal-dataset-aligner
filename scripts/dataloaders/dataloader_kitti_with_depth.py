import os
from torch.utils.data import Dataset
import PIL.Image
import torch
import numpy as np
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch.nn.functional as F

import cv2 as cv

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(mode=True, warn_only=True)

Default_lat = 49.015
Satmap_zoom = 18

GrdImg_H = 364  
GrdImg_W = 1232  
GrdOriImg_H = 375
GrdOriImg_W = 1242
SatMap_original_sidelength = 512 
SatMap_process_sidelength = 504 

satmap_dir = 'satmap'
grdimage_dir = 'raw_data'
oxts_dir = 'oxts/data'  
left_color_camera_dir = 'image_02/data'
CameraGPS_shift_left = [1.08, 0.26]

satmap_transform = transforms.Compose([
        # transforms.Resize(size=[SatMap_process_sidelength, SatMap_process_sidelength]),
        transforms.ToTensor()
    ])

grdimage_transform = transforms.Compose([
        transforms.Resize(size=[GrdImg_H, GrdImg_W]),
        transforms.ToTensor()
    ])

def get_meter_per_pixel(lat=Default_lat, zoom=Satmap_zoom, scale=SatMap_process_sidelength/SatMap_original_sidelength):
    meter_per_pixel = 156543.03392 * np.cos(lat * np.pi/180.) / (2**zoom)	
    meter_per_pixel /= 2 # because use scale 2 to get satmap 
    meter_per_pixel /= scale
    return meter_per_pixel

class SatGrdDataset(Dataset):
    def __init__(self, root, file,
                 transform=None, shift_range_lat=20, shift_range_lon=20, rotation_range=10):
        self.root = root

        self.meter_per_pixel = get_meter_per_pixel(scale=1)
        self.shift_range_meters_lat = shift_range_lat  # in terms of meters
        self.shift_range_meters_lon = shift_range_lon  # in terms of meters
        self.shift_range_pixels_lat = shift_range_lat / self.meter_per_pixel  # shift range is in terms of pixels
        self.shift_range_pixels_lon = shift_range_lon / self.meter_per_pixel  # shift range is in terms of pixels

        self.rotation_range = rotation_range  # in terms of degree

        self.skip_in_seq = 2  # skip 2 in sequence: 6,3,1~
        if transform != None:
            self.satmap_transform = transform[0]
            self.grdimage_transform = transform[1]

        self.pro_grdimage_dir = 'raw_data'

        self.satmap_dir = satmap_dir

        with open(file, 'r') as f:
            file_name = f.readlines()

        self.file_name = [file[:-1] for file in file_name]

    def __len__(self):
        return len(self.file_name)

    def get_file_list(self):
        return self.file_name

    def _load_depth(self, path):
        try:
            inv_depth = np.load(path) 

            depth = 1.0 / (inv_depth + 1e-10) 
            metric_depth = np.clip(metric_depth, 0, 40.0)
            
            return depth
        except Exception as e:
            print('check')
            print(f'Unreadable depth image: {path} ({e})')
        return None

    def _load_metric_depth(self, path):
        try:
            metric_depth = cv.imread(path, cv.IMREAD_UNCHANGED)
            metric_depth = metric_depth.astype(np.float32) / 256.0
            metric_depth = np.clip(metric_depth, 0, 40.0)

            return metric_depth
        except Exception as e:
            print('check')
            print(f'Unreadable depth image: {path} ({e})')
        return None
    
    def __getitem__(self, idx):
        # read cemera k matrix from camera calibration files, day_dir is first 10 chat of file name

        file_name = self.file_name[idx]
        day_dir = file_name[:10]
        drive_dir = file_name[:38]#2011_09_26/2011_09_26_drive_0002_sync/
        image_no = file_name[38:]

        # =================== read camera intrinsice for left and right cameras ====================
        calib_file_name = os.path.join(self.root, grdimage_dir, day_dir, 'calib_cam_to_cam.txt')
        with open(calib_file_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # left color camera k matrix
                if 'P_rect_02' in line:
                    # get 3*3 matrix from P_rect_**:
                    items = line.split(':')
                    valus = items[1].strip().split(' ')
                    fx = float(valus[0]) * GrdImg_W / GrdOriImg_W
                    cx = float(valus[2]) * GrdImg_W / GrdOriImg_W
                    fy = float(valus[5]) * GrdImg_H / GrdOriImg_H
                    cy = float(valus[6]) * GrdImg_H / GrdOriImg_H
                    left_camera_k = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                    left_camera_k = torch.from_numpy(np.asarray(left_camera_k, dtype=np.float32))
                    # if not self.stereo:
                    break
        
        # =================== read satellite map ===================================
        SatMap_name = os.path.join(self.root, self.satmap_dir, file_name)
        with PIL.Image.open(SatMap_name, 'r') as SatMap:
            sat_map = SatMap.convert('RGB')

        # =================== initialize some required variables ============================
        grd_left_imgs = torch.tensor([])
        depth = torch.tensor([])
        image_no = file_name[38:]

        # oxt: such as 0000000000.txt
        oxts_file_name = os.path.join(self.root, grdimage_dir, drive_dir, oxts_dir,
                                      image_no.lower().replace('.png', '.txt'))
        with open(oxts_file_name, 'r') as f:
            content = f.readline().split(' ')
            # get heading
            lat = float(content[0])
            lon = float(content[1])
            heading = float(content[5])

            left_img_name = os.path.join(self.root, self.pro_grdimage_dir, drive_dir, left_color_camera_dir,
                                         image_no.lower())
            
            inv_depth_name = left_img_name.replace('raw_data', 'depth_anythingv2_invdepth').replace('png', 'npy')
            metric_depth_name = left_img_name.replace('raw_data', 'depth_anythingv2_depth').replace('image_02/data', 'depth')
            
            with PIL.Image.open(left_img_name, 'r') as GrdImg:
                grd_img_left = GrdImg.convert('RGB')
                if self.grdimage_transform is not None:
                    grd_img_left = self.grdimage_transform(grd_img_left)

            grd_left_imgs = torch.cat([grd_left_imgs, grd_img_left.unsqueeze(0)], dim=0)

            # depth = self._load_depth(inv_depth_name)
            depth = self._load_metric_depth(metric_depth_name)
            
            depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0) 
            depth = F.interpolate(depth, size=(GrdImg_H, GrdImg_W), mode='nearest').squeeze(0)

        random_scale = np.random.uniform(0.1, 10)
        random_scale = 1
        depth = depth * random_scale
        
        sat_rot = sat_map.rotate((-heading) / np.pi * 180) # make the east direction the vehicle heading
        sat_align_cam = sat_rot.transform(sat_rot.size, PIL.Image.AFFINE,
                                          (1, 0, CameraGPS_shift_left[0] / self.meter_per_pixel,
                                           0, 1, CameraGPS_shift_left[1] / self.meter_per_pixel),
                                          resample=PIL.Image.BILINEAR) 
        # randomly generate shift
        gt_shift_x = np.random.uniform(-1, 1)  # --> right as positive, parallel to the heading direction
        gt_shift_y = np.random.uniform(-1, 1)  # --> up as positive, vertical to the heading direction
        
        sat_rand_shift = \
            sat_align_cam.transform(
                sat_align_cam.size, PIL.Image.AFFINE,
                (1, 0, gt_shift_x * self.shift_range_pixels_lon,
                 0, 1, -gt_shift_y * self.shift_range_pixels_lat),
                resample=PIL.Image.BILINEAR)
        
        # randomly generate roation
        random_ori = np.random.uniform(-1, 1) * self.rotation_range # 0 means the arrow in aerial image heading Easting, counter-clockwise increasing for satellite rotation
        
        sat_rand_shift_rand_rot = sat_rand_shift.rotate(random_ori)
        
        sat_map =TF.center_crop(sat_rand_shift_rand_rot, SatMap_process_sidelength)
        
        # transform
        if self.satmap_transform is not None:
            sat_map = self.satmap_transform(sat_map)

        # location gt
        x_offset = int(gt_shift_x*self.shift_range_pixels_lon*np.cos(random_ori/180*np.pi) - gt_shift_y*self.shift_range_pixels_lat*np.sin(random_ori/180*np.pi)) # horizontal direction
        y_offset = int(-gt_shift_y*self.shift_range_pixels_lat*np.cos(random_ori/180*np.pi) - gt_shift_x*self.shift_range_pixels_lon*np.sin(random_ori/180*np.pi)) # vertical direction
        
        x_offset = x_offset/SatMap_original_sidelength*SatMap_process_sidelength
        y_offset = y_offset/SatMap_original_sidelength*SatMap_process_sidelength
        
        # orientation gt
        orientation_angle = 90 - random_ori # from ground ori (assuming heading north) to up in satellite clockwise increasing

        yaw = -orientation_angle * (np.pi/180)
        
        gt_loc = torch.tensor([[y_offset, x_offset]]) 
        gt_loc = -gt_loc # make it from grd to sat

        r = np.zeros((2,2), dtype=np.float32)
        r[0,0] = np.cos(yaw)
        r[0,1] = -np.sin(yaw)
        r[1,0] = np.sin(yaw) 
        r[1,1] = np.cos(yaw)
        r = torch.tensor(r)
        
        return sat_map, grd_left_imgs[0], depth, left_camera_k, gt_loc, r, 1/random_scale
               
class SatGrdDatasetTest(Dataset):
    def __init__(self, root, file,
                 transform=None, shift_range_lat=20, shift_range_lon=20, rotation_range=10):
        self.root = root

        self.meter_per_pixel = get_meter_per_pixel(scale=1)
        self.shift_range_meters_lat = shift_range_lat  # in terms of meters
        self.shift_range_meters_lon = shift_range_lon  # in terms of meters
        self.shift_range_pixels_lat = shift_range_lat / self.meter_per_pixel  # shift range is in terms of meters
        self.shift_range_pixels_lon = shift_range_lon / self.meter_per_pixel  # shift range is in terms of meters

        self.rotation_range = rotation_range  # in terms of degree

        self.skip_in_seq = 2  # skip 2 in sequence: 6,3,1~
        if transform != None:
            self.satmap_transform = transform[0]
            self.grdimage_transform = transform[1]

        self.pro_grdimage_dir = 'raw_data'

        self.satmap_dir = satmap_dir

        with open(file, 'r') as f:
            file_name = f.readlines()

        self.file_name = [file[:-1] for file in file_name]
       

    def __len__(self):
        return len(self.file_name)

    def get_file_list(self):
        return self.file_name

    def _load_depth(self, path):
        try:
            inv_depth = np.load(path) 

            depth = 1.0 / (inv_depth + 1e-10) 

            return depth
        except Exception as e:
            print('check')
            print(f'Unreadable depth image: {path} ({e})')
        return None
    
    def _load_metric_depth(self, path):
        try:
            metric_depth = cv.imread(path, cv.IMREAD_UNCHANGED)
            metric_depth = metric_depth.astype(np.float32) / 256.0
            metric_depth = np.clip(metric_depth, 0, 40.0)

            return metric_depth
        except Exception as e:
            print('check')
            print(f'Unreadable depth image: {path} ({e})')
        return None

    def __getitem__(self, idx):

        line = self.file_name[idx]
        file_name, gt_shift_x, gt_shift_y, theta = line.split(' ')
        day_dir = file_name[:10]
        drive_dir = file_name[:38]
        image_no = file_name[38:]

        # =================== read camera intrinsice for left and right cameras ====================
        calib_file_name = os.path.join(self.root, grdimage_dir, day_dir, 'calib_cam_to_cam.txt')
        with open(calib_file_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # left color camera k matrix
                if 'P_rect_02' in line:
                    # get 3*3 matrix from P_rect_**:
                    items = line.split(':')
                    valus = items[1].strip().split(' ')
                    fx = float(valus[0]) * GrdImg_W / GrdOriImg_W
                    cx = float(valus[2]) * GrdImg_W / GrdOriImg_W
                    fy = float(valus[5]) * GrdImg_H / GrdOriImg_H
                    cy = float(valus[6]) * GrdImg_H / GrdOriImg_H
                    left_camera_k = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                    left_camera_k = torch.from_numpy(np.asarray(left_camera_k, dtype=np.float32))
                    # if not self.stereo:
                    break
        
        
        # =================== read satellite map ===================================
        SatMap_name = os.path.join(self.root, self.satmap_dir, file_name)
        with PIL.Image.open(SatMap_name, 'r') as SatMap:
            sat_map = SatMap.convert('RGB')

        # =================== initialize some required variables ============================
        grd_left_imgs = torch.tensor([])
        grd_left_depths = torch.tensor([])
        # image_no = file_name[38:]

        # oxt: such as 0000000000.txt
        oxts_file_name = os.path.join(self.root, grdimage_dir, drive_dir, oxts_dir,
                                      image_no.lower().replace('.png', '.txt'))
        with open(oxts_file_name, 'r') as f:
            content = f.readline().split(' ')
            # get heading
            lat = float(content[0])
            lon = float(content[1])
            heading = float(content[5])

            left_img_name = os.path.join(self.root, self.pro_grdimage_dir, drive_dir, left_color_camera_dir,
                                         image_no.lower())
            
            inv_depth_name = left_img_name.replace('raw_data', 'depth_anythingv2_invdepth').replace('png', 'npy')
            metric_depth_name = left_img_name.replace('raw_data', 'depth_anythingv2_depth').replace('image_02/data', 'depth')
            
            with PIL.Image.open(left_img_name, 'r') as GrdImg:
                grd_img_left = GrdImg.convert('RGB')
                if self.grdimage_transform is not None:
                    grd_img_left = self.grdimage_transform(grd_img_left)

            grd_left_imgs = torch.cat([grd_left_imgs, grd_img_left.unsqueeze(0)], dim=0)
            
            # depth = self._load_depth(inv_depth_name)
            depth = self._load_metric_depth(metric_depth_name)
            
            depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0) 
            depth = F.interpolate(depth, size=(GrdImg_H, GrdImg_W), mode='nearest').squeeze(0)

        
        sat_rot = sat_map.rotate(-heading / np.pi * 180)
        
        sat_align_cam = sat_rot.transform(sat_rot.size, PIL.Image.AFFINE,
                                          (1, 0, CameraGPS_shift_left[0] / self.meter_per_pixel,
                                           0, 1, CameraGPS_shift_left[1] / self.meter_per_pixel),
                                          resample=PIL.Image.BILINEAR)
        
        # load the shifts 
        gt_shift_x = -float(gt_shift_x)  # --> right as positive, parallel to the heading direction
        gt_shift_y = -float(gt_shift_y)  # --> up as positive, vertical to the heading direction

        sat_rand_shift = \
            sat_align_cam.transform(
                sat_align_cam.size, PIL.Image.AFFINE,
                (1, 0, gt_shift_x * self.shift_range_pixels_lon,
                 0, 1, -gt_shift_y * self.shift_range_pixels_lat),
                resample=PIL.Image.BILINEAR)
        random_ori = float(theta) * self.rotation_range # degree

        sat_rand_shift_rand_rot = sat_rand_shift.rotate(random_ori)
        
        sat_map =TF.center_crop(sat_rand_shift_rand_rot, SatMap_process_sidelength)


        # transform
        if self.satmap_transform is not None:
            sat_map = self.satmap_transform(sat_map)

        # location gt
        x_offset = (gt_shift_x*self.shift_range_pixels_lon*np.cos(random_ori/180*np.pi) - gt_shift_y*self.shift_range_pixels_lat*np.sin(random_ori/180*np.pi)) # horizontal direction
        y_offset = (-gt_shift_y*self.shift_range_pixels_lat*np.cos(random_ori/180*np.pi) - gt_shift_x*self.shift_range_pixels_lon*np.sin(random_ori/180*np.pi)) # vertical direction
        
        x_offset = x_offset/SatMap_original_sidelength*SatMap_process_sidelength
        y_offset = y_offset/SatMap_original_sidelength*SatMap_process_sidelength

        gt_loc = torch.tensor([[y_offset, x_offset]], dtype=torch.float32)
        gt_loc = -gt_loc # make it from grd to sat

        # orientation gt
        orientation_angle = 90 - random_ori # from ground ori (assuming heading north) to satellite ori clockwise increasing

        yaw = -orientation_angle * (np.pi/180) 
        

        r = np.zeros((2,2), dtype=np.float32)
        r[0,0] = np.cos(yaw)
        r[0,1] = -np.sin(yaw)
        r[1,0] = np.sin(yaw) 
        r[1,1] = np.cos(yaw)
        r = torch.tensor(r)
        
        scale = 1
        return sat_map, grd_left_imgs[0], depth, left_camera_k, gt_loc, r, scale