import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--machine', type=str, help='scitas or local', default='local')
parser.add_argument('-l', '--learning_rate', type=float, help='learning rate', default=1e-4)
parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=4)
parser.add_argument('--beta', type=float, help='weight on the infoNCE loss', default=1e0)
parser.add_argument('--epoch_to_resume', type=int, help='epoch number to resume training, will load checkpoint from this number minus one', default=0)
parser.add_argument('--name', type=str, help='name of the experiment', default='')
parser.add_argument('--training_ratio', type=float, help='ratio to split training and validation data', default=0.7)

args = vars(parser.parse_args())
machine = args['machine']
learning_rate = args['learning_rate']
beta = args['beta']
epoch_to_resume = args['epoch_to_resume']
experiment_name = args['name']

# Load configuration
import configparser
config = configparser.ConfigParser()
config.read("./config.ini")

import torch
import numpy as np
import random
from dataloaders.dataloader_RGBT_Scenes import RGBT_Scenes_Dataset, random_split
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F

from models.model_RGBT import CVM_Thermal
from models.model_RGBT_simple import CVM_Thermal_Simple
from models.modules import DinoExtractor

from keypoint_patches import compute_patch_matches, visualize_patch_matches

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

# set_seeds(config.getint("RandomSeed", "seed"))

# Load hyperparameters from config

if machine == 'local':
    dataset_root = config["RGBT_Scenes"]["local_dataset_root"]
    learning_rate = config.getfloat("Training", "learning_rate")
    epoch_to_resume = epoch_to_resume if epoch_to_resume > 0 else config.getint("Training", "epoch_to_resume")
    beta = config.getfloat("Loss", "beta")

elif machine == 'scitas':
    dataset_root = config["RGBT_Scenes"]["scitas_dataset_root"]

grid_size_h = config.getfloat("RGBT_Scenes", "grid_size_h")
grid_size_v = config.getfloat("RGBT_Scenes", "grid_size_v")

grd_bev_res = config.getint("Model", "grd_bev_res")
grd_height_res = config.getint("Model", "grd_height_res")
sat_bev_res = config.getint("Model", "sat_bev_res")

num_keypoints = config.getint("Model", "num_keypoints")
num_samples_matches = config.getint("Matching", "num_samples_matches")

loss_grid_size = config.getfloat("Loss", "loss_grid_size")
num_virtual_point = config.getint("Loss", "num_virtual_point")


# Load dataset
print(dataset_root)

# Create DataLoaders
train_data, val_data = RGBT_Scenes_Dataset.build_test_train_dataloaders(dataset_root, training_ratio=args['training_ratio'], low_memory_mode=True)
print(len(train_data), 'training samples found.')
print(len(val_data), 'validation samples found.')

train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4, collate_fn=train_data.collate_fn)
val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4, collate_fn=val_data.collate_fn)
# val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
# test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

# Print the indices of each split if desired for reproducibility
with open('train_indices.txt', 'w') as f:
    f.write('\n'.join(map(str, train_data.thermal_images)))
with open('val_indices.txt', 'w') as f:
    f.write('\n'.join(map(str, val_data.thermal_images)))

# Free unused memory
# torch.cuda.empty_cache()

# Initialize shared feature extractor
shared_feature_extractor = DinoExtractor().to(device)

# Initialize CVM Model
# CVM_model = CVM_Thermal(device, grd_bev_res=grd_bev_res, grd_height_res=grd_height_res, 
#                 sat_bev_res=sat_bev_res, num_keypoints=num_keypoints, 
#                 embed_dim=1024, grid_size_h=grid_size_h, grid_size_v=grid_size_v)
CVM_model = CVM_Thermal_Simple(device, num_keypoints=num_keypoints, temperature=0.1, embed_dim=1024, desc_dim=128)

# Generate label for logging
label = (f"RGBT_Scenes_{CVM_model.__class__.__name__}_num_matches_{num_samples_matches}"
         f"_beta_{beta}_grd_bev_res_{grd_bev_res}_height_res_{grd_height_res}"
         f"_sat_res_{sat_bev_res}_loss_grid_{loss_grid_size}"
         f"_h_{int(grid_size_h)}_v_{grid_size_v}_lr_{learning_rate}_name_{experiment_name}")

print(f"Experiment label: {label}")

global_step = 0

# Load checkpoint if resuming training
if epoch_to_resume > 0:
    print("Resuming training from epoch", epoch_to_resume)
    model_path = f'../checkpoints/{label}/{epoch_to_resume}/model.pt'
    global_step, rng_state, state_dict = torch.load(model_path)
    print(f"Loaded checkpoint from {model_path} at global step {global_step}.")
    CVM_model.load_state_dict(state_dict)
    torch.set_rng_state(rng_state)
    
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

# -------------------------
# Training Loop
# -------------------------
print(f'📊 Model: {label}. Evaluating on validation set...')
results_dir = '../results/'+label+'/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

with torch.no_grad():
    CVM_model.eval()
    distance_error = []

    # visualization_index = torch.randint(0, len(val_dataloader), (1,)).item()
    visualization_index = 0
    for i, data in enumerate(val_dataloader, 0):
        img1, img2, conf, patches_1, patches_2, patches_1_len, patches_2_len = data

        img1, img2, patches_1, patches_2, patches_1_len, patches_2_len = img1.to(device), img2.to(device), patches_1.long().to(device), patches_2.long().to(device), patches_1_len.to(device), patches_2_len.to(device)

        img1_feature, img2_feature = shared_feature_extractor(img1), shared_feature_extractor(img2)

        matching_score, img2_indices_topk, img1_indices_topk, matching_score_original = CVM_model(img1_feature, img2_feature)
        # _, num_kpts_sat, num_kpts_grd = matching_score.shape
        # patches_1, patches_2 = compute_patch_matches(kpts1.squeeze(0), kpts2.squeeze(0), img1.shape[2:], patch_size=14)
        max_keypoints_for_comparison = min(num_keypoints, patches_1.shape[0], patches_2.shape[0])
        
        max_len = max(patches_1_len.max().item(), patches_2_len.max().item())
        range_tensor = torch.arange(max_len).unsqueeze(0).to(device)  # shape (1, max_len)
        patches_1_mask = range_tensor < patches_1_len.unsqueeze(1)  # shape (batch_size, max_len)
        patches_2_mask = range_tensor < patches_2_len.unsqueeze(1)  # shape (batch_size, max_len)
        max_num_keypoint_mask = torch.full_like(patches_1_mask, False).to(device)
        max_num_keypoint_mask[:, :num_keypoints] = True
        max_keypoints_mask = patches_1_mask & max_num_keypoint_mask & patches_2_mask

        B = img1.shape[0]
        batch_idx = torch.arange(B).unsqueeze(1).to(device)  # shape (B,1)
        scores_img1 = matching_score[batch_idx, patches_2]   # (batch, number of gt matches, number of patch categories)
        scores_img2 = matching_score.transpose(1,2)[batch_idx, patches_1]  # (batch, number of gt matches, number of patch categories)
        
        # if i == visualization_index:
        for item_to_pick in range(B):
            visualize_patch_matches(img1[item_to_pick].permute(1,2,0).cpu().numpy(), img2[item_to_pick].permute(1,2,0).cpu().numpy(), list(zip(img1_indices_topk[item_to_pick].reshape(num_keypoints,).cpu().numpy(), img2_indices_topk[item_to_pick].reshape(num_keypoints,).cpu().numpy())), patch_size=14, patches_to_draw=10)

        img1_loss = F.cross_entropy(
            scores_img1[max_keypoints_mask],
            patches_1[max_keypoints_mask]
        )
        # img1_topk_loss = F.cross_entropy(
        #     matching_score.squeeze(0)[img1_indices_topk.squeeze(0), :][:max_keypoints_for_comparison, :],
        #     patches_1.squeeze(0)[:max_keypoints_for_comparison].to(device)
        # )
        img2_loss = F.cross_entropy(
            scores_img2[max_keypoints_mask],
            patches_2[max_keypoints_mask]
        )
        # img2_topk_loss = F.cross_entropy(
        #     matching_score.squeeze(0).transpose(1, 0)[img2_indices_topk.squeeze(0), :][:max_keypoints_for_comparison, :],
        #     patches_2.squeeze(0)[:max_keypoints_for_comparison].to(device)
        # )
        distance_loss = img1_loss + img2_loss # + img1_topk_loss + img2_topk_loss
        distance_error.append(distance_loss.item())       
    
    distance_error_mean = np.mean(distance_error)    
    distance_error_median = np.median(distance_error) 

    print(f'📉 Mean Distance Error: {distance_error_mean:.3f}')
    print(f'📉 Median Distance Error: {distance_error_median:.3f}')
