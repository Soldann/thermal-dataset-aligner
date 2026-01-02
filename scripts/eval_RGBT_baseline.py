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
parser.add_argument('--model_type', type=str, help='type of model to use: xoftr or cvm_simple', default='xoftr')

args = vars(parser.parse_args())
machine = args['machine']
learning_rate = args['learning_rate']
beta = args['beta']
epoch_to_resume = args['epoch_to_resume']
experiment_name = args['name']
model_type = args['model_type']

# Load configuration
import configparser
config = configparser.ConfigParser()
config.read("./config.ini")

import torch
import numpy as np
import random
import cv2
import ahrs
from dataloaders.dataloader_RGBT_Scenes import RGBT_Scenes_Dataset, random_split
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F

from models.model_RGBT import CVM_Thermal
from models.model_RGBT_simple import CVM_Thermal_Simple
from models.model_xoftr import ModelXoFTR
from models.model_loftr import ModelLoFTR
from models.model_match_anything import ModelMatchAnything
from models.modules import DinoExtractor

from keypoint_patches import compute_patch_matches, visualize_patch_matches, convert_patches_to_keypoints

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
if model_type == 'cvm_simple':
    CVM_model = CVM_Thermal_Simple(device, num_keypoints=num_keypoints, temperature=0.1, embed_dim=1024, desc_dim=128)
elif model_type == 'xoftr':
    CVM_model = ModelXoFTR(device)
elif model_type == 'loftr':
    CVM_model = ModelLoFTR(device)
elif model_type == 'match_anything':
    CVM_model = ModelMatchAnything(device)
else:
    raise ValueError(f"Unknown model type: {model_type}")
CVM_model = CVM_model.to(device)

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

# -------------------------
# Training Loop
# -------------------------
print(f'📊 Model: {label}. Evaluating on validation set...')
results_dir = '../results/'+label+'/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

with torch.no_grad():
    CVM_model.eval()
    val_error = []
    overall_rotation_errors = []
    overall_translation_errors = []

    # visualization_index = torch.randint(0, len(val_dataloader), (1,)).item()
    visualization_index = 0
    for i, data in enumerate(val_dataloader, 0):
        img1, img2, conf, patches_1, patches_2, patches_1_len, patches_2_len, img1_pose, img2_inv_pose, img1_K, img2_K = data

        img1, img2, patches_1, patches_2, patches_1_len, patches_2_len, img1_pose, img2_inv_pose, img1_K, img2_K = img1.to(device), img2.to(device), patches_1.long().to(device), patches_2.long().to(device), patches_1_len.to(device), patches_2_len.to(device), img1_pose.to(device), img2_inv_pose.to(device), img1_K.to(device), img2_K.to(device)
            
        B = img1.shape[0]
        # Compute pose error
        homogenous_row = torch.tensor([0,0,0,1], dtype=torch.float32).view(1,1,4).repeat(B,1,1).to(device)
        c1Tw = torch.cat([img1_pose, homogenous_row], dim=1) # world to cam1
        wTc2 = torch.cat([img2_inv_pose, homogenous_row], dim=1) # cam2 to world
        c1Tc2 = torch.bmm(c1Tw, wTc2) # cam2 to cam1
        
        rotation_errors = []
        translation_errors = []
        for batch_item in range(B):
            if model_type == 'cvm_simple':
                img1_feature, img2_feature = shared_feature_extractor(img1[batch_item].unsqueeze(0)), shared_feature_extractor(img2[batch_item].unsqueeze(0))
                matching_score, img2_indices_topk, img1_indices_topk, matching_score_original = CVM_model(img1_feature, img2_feature)

                # max_keypoints_for_comparison = min(num_keypoints, patches_1.shape[0], patches_2.shape[0])
                    
                # max_len = max(patches_1_len.max().item(), patches_2_len.max().item())
                # range_tensor = torch.arange(max_len).unsqueeze(0).to(device)  # shape (1, max_len)
                # patches_1_mask = range_tensor < patches_1_len.unsqueeze(1)  # shape (batch_size, max_len)
                # patches_2_mask = range_tensor < patches_2_len.unsqueeze(1)  # shape (batch_size, max_len)
                # max_num_keypoint_mask = torch.full_like(patches_1_mask, False).to(device)
                # max_num_keypoint_mask[:, :num_keypoints] = True
                # max_keypoints_mask = patches_1_mask & max_num_keypoint_mask & patches_2_mask

                kpts1 = convert_patches_to_keypoints(img1_indices_topk.squeeze(0).reshape(num_keypoints,), img1.shape[2:]).cpu().numpy()
                kpts2 = convert_patches_to_keypoints(img2_indices_topk.squeeze(0).reshape(num_keypoints,), img2.shape[2:]).cpu().numpy()
            elif model_type == 'xoftr' or model_type == 'loftr' or model_type == 'match_anything':
                kpts1, kpts2 = CVM_model(img1[batch_item], img2[batch_item])

            E, mask = cv2.findEssentialMat(kpts1, kpts2, img1_K[batch_item].cpu().numpy(), method=cv2.RANSAC, prob=0.999, threshold=1.0)
            mask = mask.astype(bool).reshape(-1)
            _, R_est, t_est, mask_pose = cv2.recoverPose(E, kpts1[mask], kpts2[mask], cameraMatrix=img1_K[batch_item].cpu().numpy())

            # compute error in pose
            R_gt = c1Tc2[batch_item, :3, :3]
            t_gt = c1Tc2[batch_item, :3, 3]

            # R_est = torch.tensor(R_est).to(device)
            t_est = torch.tensor(t_est).to(device)
            t_est = t_est / torch.norm(t_est) * torch.norm(t_gt) # set scale of the estimated translation to be the same as ground truth 

            # Compute translation error
            translation_errors.extend(torch.norm(t_est - t_gt, dim=-1).cpu().numpy())

            # Compute yaw error
            Rgt_np, R_np = R_gt.cpu().numpy(), R_est

            quaternion_angle_diff = ahrs.utils.metrics.qad(ahrs.Quaternion(dcm=R_np), ahrs.Quaternion(dcm=Rgt_np))
            rotation_errors.append(quaternion_angle_diff)   

        overall_rotation_errors.extend(rotation_errors)
        overall_translation_errors.extend(translation_errors)
    
    rotation_error_mean = np.mean(overall_rotation_errors)
    rotation_error_median = np.median(overall_rotation_errors)
    translation_error_mean = np.mean(overall_translation_errors)
    translation_error_median = np.median(overall_translation_errors)

    print(f'Mean Rotation Error: {rotation_error_mean:.3f}')
    print(f'Median Rotation Error: {rotation_error_median:.3f}')
    print(f'Mean Translation Error: {translation_error_mean:.3f}')
    print(f'Median Translation Error: {translation_error_median:.3f}')
