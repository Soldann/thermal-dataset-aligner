import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--learning_rate', type=float, help='learning rate', default=1e-4)
parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=80)
parser.add_argument('--rotation_range', type=float, help='random orientation range from -x degrees to x degrees', default=10)
parser.add_argument('--beta', type=float, help='weight on the infoNCE loss', default=1e0)
parser.add_argument('--epoch_to_resume', type=int, help='epoch number to resume training, will load checkpoint from this number minus one', default=0)
parser.add_argument('--loss_grid_size', type=float, help='grid size for metric coordinates to calcualte loss', default=5)
parser.add_argument('--estimate_scale', choices=('True','False'), default='True', help='If we estimate the scale in Procrustes (True/False)')
parser.add_argument('--initial_scale', type=float, help='initial scale of the depth map', default=1)
parser.add_argument('--max_depth', type=float, default=40)
parser.add_argument('--scale_loss_weight', type=float, help='weight on the scale loss', default=0)



args = vars(parser.parse_args())
learning_rate = args['learning_rate']
batch_size = args['batch_size']
rotation_range = args['rotation_range']
beta = args['beta']
scale_loss_weight = args['scale_loss_weight']
epoch_to_resume = args['epoch_to_resume']
estimate_scale = args['estimate_scale'] == 'True'
loss_grid_size = args['loss_grid_size']
# initial_scale = args['initial_scale']
max_depth = args['max_depth']

# Load configuration
import configparser
config = configparser.ConfigParser()
config.read("./config.ini")
import ast

import torch
import numpy as np
import random
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, default_collate
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import math

from dataloaders.dataloader_kitti_with_depth import SatGrdDataset, SatGrdDatasetTest, grdimage_transform, satmap_transform
from utils.utils import weighted_procrustes_2d, weighted_procrustes_2d_with_scale
from models.loss import loss_bev_space, compute_infonce_loss_match_all, compute_infonce_loss_match_all_with_scale, scale_loss_log_l1, compute_infonce_loss_match_all_with_scale_select_negatives
from models.model_kitti_depth_matchall import CVM
from models.modules import DinoExtractor

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility from config
def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(mode=True, warn_only=True)

set_seeds(config.getint("RandomSeed", "seed"))
epsilon = config.getfloat("Constants", "epsilon")

# Load hyperparameters from config

dataset_root = config["KITTI"]["local_dataset_root"]
set_seeds(config.getint("RandomSeed", "seed"))
epsilon = config.getfloat("Constants", "epsilon")

shift_range_lat = config.getfloat("KITTI", "shift_range_lat")
shift_range_lon = config.getfloat("KITTI", "shift_range_lon")

grid_size_h = config.getfloat("KITTI", "grid_size_h")
sat_bev_res = config.getint("Model", "sat_bev_res")

num_samples_matches = config.getint("Matching", "num_samples_matches")
num_virtual_point = config.getint("Loss", "num_virtual_point")

# Load image size from config
GrdImg_H = config.getint("KITTI", "GrdImg_H")
GrdImg_W = config.getint("KITTI", "GrdImg_W")
GrdOriImg_H = config.getint("KITTI", "GrdOriImg_H")
GrdOriImg_W = config.getint("KITTI", "GrdOriImg_W")
SatMap_original_sidelength = config.getint("KITTI", "SatMap_original_sidelength")
SatMap_process_sidelength = config.getint("KITTI", "SatMap_process_sidelength")


# Load dataset
num_thread_workers = 4

train_file = './KITTI_splits/train_files.txt'
test1_file = './KITTI_splits/test1_files.txt'
test2_file = './KITTI_splits/test2_files.txt'


train_set = SatGrdDataset(root=dataset_root, file=train_file,
                              transform=(satmap_transform, grdimage_transform),
                              shift_range_lat=shift_range_lat,
                              shift_range_lon=shift_range_lon,
                              rotation_range=rotation_range)

test1_set = SatGrdDatasetTest(root=dataset_root, file=test1_file,
                              transform=(satmap_transform, grdimage_transform),
                              shift_range_lat=shift_range_lat,
                              shift_range_lon=shift_range_lon,
                              rotation_range=rotation_range)

test2_set = SatGrdDatasetTest(root=dataset_root, file=test2_file,
                              transform=(satmap_transform, grdimage_transform),
                              shift_range_lat=shift_range_lat,
                              shift_range_lon=shift_range_lon,
                              rotation_range=rotation_range)

# Create DataLoaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True,
                              num_workers=num_thread_workers, drop_last=False)

test1_loader = DataLoader(test1_set, batch_size=batch_size, shuffle=False, pin_memory=True,
                            num_workers=num_thread_workers, drop_last=False)

test2_loader = DataLoader(test2_set, batch_size=batch_size, shuffle=False, pin_memory=True,
                              num_workers=num_thread_workers, drop_last=False)


# Generate label for logging
# label = (
#     f"KITTI_depth_rotation_range_{rotation_range}"
#     f"_num_samples_matches_{num_samples_matches}"
#     f"_beta_{beta}_loss_grid_size_{loss_grid_size}"
#     f"_sat_res_{sat_bev_res}_estimate_scale_{estimate_scale}"
#     f"_initial_scale_{initial_scale}_max_depth_{max_depth}"
# )

label = (
    f"KITTI_metricdepth_infonceselectneg_rotation_range_{rotation_range}"
    f"_num_samples_matches_{num_samples_matches}"
    f"_beta_{beta}_loss_grid_size_{loss_grid_size}"
    f"_sat_res_{sat_bev_res}_estimate_scale_{estimate_scale}"
)

print(f"Experiment label: {label}")

# Free unused memory
torch.cuda.empty_cache()

# Initialize shared feature extractor
shared_feature_extractor = DinoExtractor(use_smaller_model=True).to(device)

# Initialize CVM Model
CVM_model = CVM(device, embed_dim=384)

# Load checkpoint if resuming training
if epoch_to_resume > 0:
    model_path = f'/home/landson/sem-project/checkpoints/{label}/{epoch_to_resume-1}/model.pt'
    CVM_model.load_state_dict(torch.load(model_path))
    
CVM_model.to(device)

# Enable gradient updates for model parameters
for param in CVM_model.parameters():
    param.requires_grad = True

# Set up optimizer
params = [p for p in CVM_model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=learning_rate, betas=(0.9, 0.999))

# Setup TensorBoard logging
writer_dir = f'../tensorboard/{label}/'
os.makedirs(writer_dir, exist_ok=True)
writer = SummaryWriter(log_dir=writer_dir)

global_step = 0

# Define metric grids for training
def create_metric_grid(grid_size, res, batch_size):
    x, y = np.linspace(-grid_size/2, grid_size/2, res), np.linspace(-grid_size/2, grid_size/2, res)
    metric_x, metric_y = np.meshgrid(x, y, indexing='ij')
    metric_x, metric_y = torch.tensor(metric_x).flatten().unsqueeze(0).unsqueeze(-1), torch.tensor(metric_y).flatten().unsqueeze(0).unsqueeze(-1)
    metric_coord = torch.cat((metric_x, metric_y), -1).to(device).float()
    return metric_coord.repeat(batch_size, 1, 1)

metric_coord_sat_B = create_metric_grid(grid_size_h, sat_bev_res, batch_size)
metric_coord4loss = create_metric_grid(loss_grid_size, num_virtual_point, 1)

u = torch.linspace(0, GrdImg_W/14, int(GrdImg_W/14)).to(device)
v = torch.linspace(0, GrdImg_H/14, int(GrdImg_H/14)).to(device)
v, u= torch.meshgrid(v, u)
v = v.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1)
u = u.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1)

# -------------------------
# Training Loop
# -------------------------
for epoch in range(epoch_to_resume, 100):
    print(f'🚀 Epoch {epoch} - Training...')
    
    running_loss = 0.0
    CVM_model.train()

    for i, data in enumerate(train_loader, 0):
        if data is None:
            continue
            
        sat, grd, depth, camera_k, tgt, Rgt, sgt = data
        B, _, sat_size, _ = sat.shape

        sat, grd, depth, camera_k, tgt, Rgt, sgt = sat.to(device), grd.to(device), depth.to(device), camera_k.to(device), tgt.to(device), Rgt.to(device), sgt.to(device)
        camera_k[:,0,:] = camera_k[:,0,:] / 14
        camera_k[:,1,:] = camera_k[:,1,:] / 14

        # # Normalize ground truth translation
        tgt = (tgt / sat_size) * grid_size_h

        # Forward pass through the feature extractor
        with torch.no_grad():
            grd_feature, sat_feature = shared_feature_extractor(grd), shared_feature_extractor(sat)

        # depth = depth * initial_scale
        # depth = torch.clip(depth, 0, max_depth)         
        depth_downsampled = F.interpolate(depth, size=grd_feature.shape[-2:], mode='nearest')
        mask = ~(depth_downsampled == max_depth / sgt.view(-1, 1, 1, 1)).flatten(1)
        
        fx, fy = camera_k[:, 0, 0], camera_k[:, 1, 1]
        cx, cy = camera_k[:, 0, 2], camera_k[:, 1, 2]

        fx = fx.view(B, 1, 1, 1)
        fy = fy.view(B, 1, 1, 1)
        cx = cx.view(B, 1, 1, 1)
        cy = cy.view(B, 1, 1, 1)
        
        grd_x = -depth_downsampled
        grd_y = (u[:B] - cx) * depth_downsampled / fx
        grd_z = (v[:B] - cy) * depth_downsampled / fy

        
        metric_coord_grd = torch.cat((grd_x.flatten(2), grd_y.flatten(2), grd_z.flatten(2)), 1).permute(0,2,1)

        bev_coord_grd = metric_coord_grd[:,:,:2]

        # Sample matching points
        matching_score, matching_score_original = CVM_model(grd_feature, sat_feature, mask)
        
        # Sample matching points
        _, num_kpts_sat, num_kpts_grd = matching_score.shape
        matches_row = matching_score.flatten(1)

        batch_idx = torch.arange(B).view(B, 1).repeat(1, num_samples_matches).reshape(B, num_samples_matches)
        sampled_matching_idx = torch.multinomial(matches_row, num_samples_matches)
 
        sat_indices_sampled = torch.div(sampled_matching_idx, num_kpts_grd, rounding_mode='trunc')
        grd_indices_sampled = sampled_matching_idx % num_kpts_grd

        exit()

        # Compute transformation using Weighted Procrustes
        X = metric_coord_sat_B[batch_idx, sat_indices_sampled, :]
        Y = bev_coord_grd[batch_idx, grd_indices_sampled, :] 
        weights = matches_row[batch_idx, sampled_matching_idx]

        if estimate_scale:
            R, t, scale, ok_rank = weighted_procrustes_2d_with_scale(Y, X, use_weights=True, use_mask=True, w=weights)
        else:
            R, t, ok_rank = weighted_procrustes_2d(Y, X, use_weights=True, use_mask=True, w=weights)

        if t is None:
            print('⚠️ Skipping batch: Singular transformation matrix')
            continue

        # # Compute distance loss
        distance_loss = torch.mean(loss_bev_space(metric_coord4loss, Rgt, tgt, R, t))

        # Compute similarity loss
        if estimate_scale:            
            # infonce_loss = torch.mean(compute_infonce_loss_match_all_with_scale(Rgt, tgt, X, Y, sgt.view(-1, 1, 1), sat_indices_sampled, grd_indices_sampled, matching_score_original, metric_coord_sat_B[:B], bev_coord_grd, mask, grid_size_h))
            # scale_loss = scale_loss_log_l1(scale, sgt)
            infonce_loss = torch.mean(compute_infonce_loss_match_all_with_scale_select_negatives(Rgt, tgt, X, Y, sgt.view(-1, 1, 1), sat_indices_sampled, grd_indices_sampled, matching_score_original, metric_coord_sat_B[:B], bev_coord_grd, mask, grid_size_h))
        else:
            infonce_loss = torch.mean(compute_infonce_loss_match_all(Rgt, tgt, X, Y, sat_indices_sampled, grd_indices_sampled, matching_score_original, metric_coord_sat_B[:B], bev_coord_grd, mask, grid_size_h))

        
        
        avg_loss = distance_loss + beta * infonce_loss #+ scale_loss_weight * scale_loss
        print(f'Epoch [{epoch}] Batch [{i}/{len(train_loader)}] - Distance Loss: {distance_loss.item():.4f}, InfoNCE Loss: {infonce_loss.item():.4f}, Avg Loss: {avg_loss.item():.4f}')
        
        # Backpropagation
        optimizer.zero_grad()
        avg_loss.backward()
        optimizer.step()

        # Log training loss
        writer.add_scalar("loss/distance", distance_loss.item(), global_step)
        writer.add_scalar("loss/infonce", infonce_loss.item(), global_step)
        writer.add_scalar("loss/avg", avg_loss.item(), global_step)
        if estimate_scale:
            writer.add_scalar("scale", (torch.mean(scale)).item(), global_step)
            # writer.add_scalar("loss/scale_loss", (torch.mean(scale_loss)).item(), global_step)
        global_step += 1

    model_dir = '../checkpoints/'+label+'/' + str(epoch) + '/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print('save checkpoint at '+model_dir)
    torch.save(CVM_model.cpu().state_dict(), model_dir+'model.pt') # saving model
    CVM_model.cuda() # moving model to GPU for further training
    
    # -------------------------
    # Validation 
    # -------------------------
    print(f'📊 Epoch {epoch} - Evaluating on validation set...')
    results_dir = '../results/'+label+'/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)     

writer.flush()
