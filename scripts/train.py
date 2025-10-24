import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--machine', type=str, help='scitas or local', default='scitas')
parser.add_argument('--area', type=str, help='samearea or crossarea', default='crossarea')
parser.add_argument('-l', '--learning_rate', type=float, help='learning rate', default=1e-4)
parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=4)
parser.add_argument('--random_orientation', type=float, help='random orientation range from -x degrees to x degrees', default=0)
parser.add_argument('--beta', type=float, help='weight on the infoNCE loss', default=1e0)
parser.add_argument('--epoch_to_resume', type=int, help='epoch number to resume training, will load checkpoint from this number minus one', default=0)

args = vars(parser.parse_args())
machine = args['machine']
area = args['area']
learning_rate = args['learning_rate']
batch_size = args['batch_size']
random_orientation = args['random_orientation']
beta = args['beta']
epoch_to_resume = args['epoch_to_resume']

# Load configuration
import configparser
config = configparser.ConfigParser()
config.read("./config.ini")

import torch
import numpy as np
import random
from dataloader_vigor import VIGORDataset, transform_grd, transform_sat
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import math

from utils import weighted_procrustes_2d
from loss import loss_bev_space, compute_infonce_loss
from model import CVM
from modules import DinoExtractor

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

# Load hyperparameters from config

if machine == 'local':
    dataset_root = config["Datasets"]["local_dataset_root"]
    batch_size = config.getint("Training", "batch_size")
    learning_rate = config.getfloat("Training", "learning_rate")
    random_orientation = config.getint("Training", "random_orientation")
    epoch_to_resume = config.getint("Training", "epoch_to_resume")
    beta = config.getfloat("Loss", "beta")

elif machine == 'scitas':
    dataset_root = config["Datasets"]["scitas_dataset_root"]
    

grd_bev_res = config.getint("Model", "grd_bev_res")
grd_height_res = config.getint("Model", "grd_height_res")
sat_bev_res = config.getint("Model", "sat_bev_res")
grid_size_h = config.getint("Model", "grid_size_h")
grid_size_v = config.getint("Model", "grid_size_v")
num_keypoints = config.getint("Model", "num_keypoints")

num_samples_matches = config.getint("Matching", "num_samples_matches")

loss_grid_size = config.getfloat("Loss", "loss_grid_size")
num_virtual_point = config.getint("Loss", "num_virtual_point")

# Load image size from config
ground_image_size = (config.getint("ImageSize", "ground_image_height"), config.getint("ImageSize", "ground_image_width"))
satellite_image_size = (config.getint("ImageSize", "satellite_image_height"), config.getint("ImageSize", "satellite_image_width"))

print(f"Ground Image Size: {ground_image_size}, Satellite Image Size: {satellite_image_size}")

# Load dataset
vigor = VIGORDataset(
    root=dataset_root, split=area, train=True, pos_only=True, 
    transform=(transform_grd, transform_sat), random_orientation=random_orientation
)

# Split dataset into training and validation sets
dataset_length = len(vigor)
indices = np.arange(dataset_length)
np.random.shuffle(indices)
split_idx = int(dataset_length * 0.8)

train_indices, val_indices = indices[:split_idx], indices[split_idx:]
training_set, val_set = Subset(vigor, train_indices), Subset(vigor, val_indices)

# Create DataLoaders
train_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

# Generate label for logging
label = (f"{area}_random_ori_{random_orientation}_num_matches_{num_samples_matches}"
         f"_beta_{beta}_grd_bev_res_{grd_bev_res}_height_res_{grd_height_res}"
         f"_sat_res_{sat_bev_res}_loss_grid_{loss_grid_size}"
         f"_h_{grid_size_h}_v_{grid_size_v}_lr_{learning_rate}")

print(f"Experiment label: {label}")

# Free unused memory
torch.cuda.empty_cache()

# Initialize shared feature extractor
shared_feature_extractor = DinoExtractor().to(device)

# Initialize CVM Model
CVM_model = CVM(device, grd_bev_res=grd_bev_res, grd_height_res=grd_height_res, 
                sat_bev_res=sat_bev_res, num_keypoints=num_keypoints, 
                embed_dim=1024, grid_size_h=grid_size_h, grid_size_v=grid_size_v)

# Load checkpoint if resuming training
if epoch_to_resume > 0:
    model_path = f'../checkpoints/{label}/{epoch_to_resume-1}/model.pt'
    CVM_model.load_state_dict(torch.load(model_path))
    
CVM_model.to(device)
CVM_model.train()

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

metric_coord_grd_B = create_metric_grid(grid_size_h, grd_bev_res, batch_size)
metric_coord_sat_B = create_metric_grid(grid_size_h, sat_bev_res, batch_size)
metric_coord4loss = create_metric_grid(loss_grid_size, num_virtual_point, 1)

# -------------------------
# Training Loop
# -------------------------
for epoch in range(epoch_to_resume, 100):
    # print(f'🚀 Epoch {epoch} - Training...')
    
    running_loss = 0.0
    CVM_model.train()

    for i, data in enumerate(train_dataloader, 0):
        grd, sat, tgt, Rgt, city = data
        B, _, sat_size, _ = sat.shape

        # Normalize ground truth translation
        tgt = (tgt / sat_size) * grid_size_h

        # Move data to device
        grd, sat, tgt, Rgt = grd.to(device), sat.to(device), tgt.to(device), Rgt.to(device)

        # Forward pass through the feature extractor
        grd_feature, sat_feature = shared_feature_extractor(grd), shared_feature_extractor(sat)     

        # Obtain matching scores and descriptors
        matching_score, sat_desc, grd_desc, sat_indices_topk, grd_indices_topk, matching_score_original = CVM_model(grd_feature, sat_feature)
        
        # Sample matching points
        _, num_kpts_sat, num_kpts_grd = matching_score.shape
        matches_row = matching_score.flatten(1)

        batch_idx = torch.arange(B).view(B, 1).repeat(1, num_samples_matches).reshape(B, num_samples_matches)
        sampled_matching_idx = torch.multinomial(matches_row, num_samples_matches)

        sampled_matching_row = torch.div(sampled_matching_idx, num_kpts_grd, rounding_mode='trunc')
        sampled_matching_col = sampled_matching_idx % num_kpts_grd

        sat_indices_sampled = torch.gather(sat_indices_topk.squeeze(1), 1, sampled_matching_row) # indices for flattened sat BEV points
        grd_indices_sampled = torch.gather(grd_indices_topk.squeeze(1), 1, sampled_matching_col) # indices for flattened grd BEV points
        
            
        # Compute transformation using Weighted Procrustes
        X, Y, weights = metric_coord_sat_B[batch_idx, sat_indices_sampled, :], metric_coord_grd_B[batch_idx, grd_indices_sampled, :], matches_row[batch_idx, sampled_matching_idx]
        R, t, ok_rank = weighted_procrustes_2d(X, Y, use_weights=True, use_mask=True, w=weights)

        if t is None:
            print('⚠️ Skipping batch: Singular transformation matrix')
            continue

        # # Compute distance loss
        distance_loss = loss_bev_space(metric_coord4loss, Rgt, tgt, R, t)

        # Compute similarity loss
        batch_idx_kpt = torch.arange(B).view(B, 1).repeat(1, num_keypoints).reshape(B, num_keypoints)
        sat_keypoint_coord, grd_keypoint_coord = metric_coord_sat_B[batch_idx_kpt, sat_indices_topk.squeeze(1), :], metric_coord_grd_B[batch_idx_kpt, grd_indices_topk.squeeze(1), :]
        infonce_loss = compute_infonce_loss(Rgt, tgt, X, Y, sampled_matching_row, sampled_matching_col, sat_indices_topk.squeeze(1), grd_indices_topk.squeeze(1), sat_keypoint_coord, grd_keypoint_coord, matching_score_original)
        
        avg_loss = torch.mean(distance_loss) + beta * torch.mean(infonce_loss)
        
        # Backpropagation
        optimizer.zero_grad()
        avg_loss.backward()
        optimizer.step()

        # Log training loss
        writer.add_scalar("Loss/train", avg_loss, global_step)
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
    
    with torch.no_grad():
        CVM_model.eval()
        translation_error, yaw_error = [], []

        for i, data in enumerate(val_dataloader, 0):
            grd, sat, tgt, Rgt, city = data
            B, _, sat_size, _ = sat.shape

            grd, sat, tgt, Rgt = grd.to(device), sat.to(device), tgt.to(device), Rgt.to(device)

            grd_feature, sat_feature = shared_feature_extractor(grd), shared_feature_extractor(sat)
    
            matching_score, sat_desc, grd_desc, sat_indices_topk, grd_indices_topk, matching_score_original = CVM_model(grd_feature, sat_feature)

            # Sample validation matches
            matches_row = matching_score.flatten(1)
            batch_idx = torch.arange(B).view(B, 1).repeat(1, num_samples_matches).reshape(B, num_samples_matches)
            sampled_matching_idx = torch.multinomial(matches_row, num_samples_matches)

            sampled_matching_row = torch.div(sampled_matching_idx, num_kpts_grd, rounding_mode='trunc')
            sampled_matching_col = sampled_matching_idx % num_kpts_grd

            sat_indices_sampled = torch.gather(sat_indices_topk.squeeze(1), 1, sampled_matching_row)
            grd_indices_sampled = torch.gather(grd_indices_topk.squeeze(1), 1, sampled_matching_col)

            X, Y, weights = metric_coord_sat_B[batch_idx, sat_indices_sampled, :], metric_coord_grd_B[batch_idx, grd_indices_sampled, :], matches_row[batch_idx, sampled_matching_idx]
            R, t, ok_rank = weighted_procrustes_2d(X, Y, use_weights=True, use_mask=True, w=weights)

            if t is None:
                print('⚠️ Skipping batch: Singular transformation matrix')
                continue
            
            # Compute translation error
            t = (t / grid_size_h) * sat_size
            translation_error.extend(torch.norm(t - tgt, dim=-1).cpu().numpy())

            # Compute yaw error
            Rgt_np, R_np = Rgt.cpu().numpy(), R.cpu().numpy()
            for b in range(B):
                cos = R_np[b,0,0]
                sin = R_np[b,1,0]
                yaw = np.degrees( np.arctan2(sin, cos) )            
                
                cos_gt = Rgt_np[b,0,0]
                sin_gt = Rgt_np[b,1,0]
                
                yaw_gt = np.degrees( np.arctan2(sin_gt, cos_gt) )
                
                diff = np.abs(yaw - yaw_gt)
    
                yaw_error.append(np.min([diff, 360-diff]))                

        translation_error_mean = np.mean(translation_error)    
        translation_error_median = np.median(translation_error)
         
        yaw_error_mean = np.mean(yaw_error)    
        yaw_error_median = np.median(yaw_error) 
    
        print('epoch:', epoch)
        print(f'📉 Mean Translation Error: {translation_error_mean:.3f}')
        print(f'📉 Median Translation Error: {translation_error_median:.3f}')
        print(f'📉 Mean Yaw Error: {yaw_error_mean:.3f}')
        print(f'📉 Median Yaw Error: {yaw_error_median:.3f}')
        
        file = results_dir+'Mean_distance_error.txt'
        with open(file,'ab') as f:
            np.savetxt(f, [translation_error_mean], fmt='%4f', header='Validation_set_mean_distance_error_in_pixels:', comments=str(epoch)+'_')
    
        file = results_dir+'Median_distance_error.txt'
        with open(file,'ab') as f:
            np.savetxt(f, [translation_error_median], fmt='%4f', header='Validation_set_median_distance_error_in_pixels:', comments=str(epoch)+'_')
    
        file = results_dir+'Mean_orientation_error.txt'
        with open(file,'ab') as f:
            np.savetxt(f, [yaw_error_mean], fmt='%4f', header='Validation_set_mean_yaw_error:', comments=str(epoch)+'_')
    
        file = results_dir+'Median_orientation_error.txt'
        with open(file,'ab') as f:
            np.savetxt(f, [yaw_error_median], fmt='%4f', header='Validation_set_median_yaw_error:', comments=str(epoch)+'_')
            
        

writer.flush()
