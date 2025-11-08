import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--machine', type=str, help='scitas or local', default='local')
parser.add_argument('-l', '--learning_rate', type=float, help='learning rate', default=1e-4)
parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=4)
parser.add_argument('--beta', type=float, help='weight on the infoNCE loss', default=1e0)
parser.add_argument('--epoch_to_resume', type=int, help='epoch number to resume training, will load checkpoint from this number minus one', default=0)

args = vars(parser.parse_args())
machine = args['machine']
learning_rate = args['learning_rate']
batch_size = args['batch_size']
beta = args['beta']
epoch_to_resume = args['epoch_to_resume']

# Load configuration
import configparser
config = configparser.ConfigParser()
config.read("./config.ini")

import torch
import numpy as np
import random
from dataloaders.dataloader_RGBT_Scenes import RGBT_Scenes_Dataset
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import math

from utils.utils import weighted_procrustes_2d
from models.loss import loss_bev_space, compute_infonce_loss
from models.model_RGBT import CVM_Thermal
from models.modules import DinoExtractor

from keypoint_patches import compute_patch_matches, visualize_patch_matches

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"  # Force CPU for debugging
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
    batch_size = config.getint("RGBT_Scenes", "batch_size")
    learning_rate = config.getfloat("Training", "learning_rate")
    epoch_to_resume = config.getint("Training", "epoch_to_resume")
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
training_set = RGBT_Scenes_Dataset(
    root=dataset_root, split='train', low_memory_mode=True
)

# val_set = RGBT_Scenes_Dataset(
#     root=dataset_root, split='val',
# )

# test_set = RGBT_Scenes_Dataset(
#     root=dataset_root, split='test',
# )

# Create DataLoaders
print(len(training_set), 'training samples found.')
train_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=4)
# val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
# test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

# Generate label for logging
label = (f"Robotcar_num_matches_{num_samples_matches}"
         f"_beta_{beta}_grd_bev_res_{grd_bev_res}_height_res_{grd_height_res}"
         f"_sat_res_{sat_bev_res}_loss_grid_{loss_grid_size}"
         f"_h_{int(grid_size_h)}_v_{grid_size_v}_lr_{learning_rate}")

print(f"Experiment label: {label}")

# Free unused memory
# torch.cuda.empty_cache()

# Initialize shared feature extractor
shared_feature_extractor = DinoExtractor().to(device)

# Initialize CVM Model
CVM_model = CVM_Thermal(device, grd_bev_res=grd_bev_res, grd_height_res=grd_height_res, 
                sat_bev_res=sat_bev_res, num_keypoints=num_keypoints, 
                embed_dim=1024, grid_size_h=grid_size_h, grid_size_v=grid_size_v)

# Load checkpoint if resuming training
if epoch_to_resume > 0:
    model_path = f'/work/vita/zimin/CVM_3D/checkpoints/{label}/{epoch_to_resume-1}/model.pt'
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
    x, y = np.linspace(-grid_size/2, grid_size/2, res[0]), np.linspace(-grid_size/2, grid_size/2, res[1])
    metric_x, metric_y = np.meshgrid(x, y, indexing='ij')
    metric_x, metric_y = torch.tensor(metric_x).flatten().unsqueeze(0).unsqueeze(-1), torch.tensor(metric_y).flatten().unsqueeze(0).unsqueeze(-1)
    metric_coord = torch.cat((metric_x, metric_y), -1).to(device).float()
    return metric_coord.repeat(batch_size, 1, 1)

metric_coord_grd_B = create_metric_grid(grid_size_h, (int(np.floor(grd_bev_res/2))+1, grd_bev_res), batch_size)
metric_coord_sat_B = create_metric_grid(grid_size_h, (sat_bev_res, sat_bev_res), batch_size)
metric_coord4loss = create_metric_grid(loss_grid_size, (num_virtual_point, num_virtual_point), 1)


# -------------------------
# Training Loop
# -------------------------
for epoch in range(epoch_to_resume, 100):
    # print(f'🚀 Epoch {epoch} - Training...')
    
    running_loss = 0.0
    CVM_model.train()

    for i, data in enumerate(train_dataloader, 0):
        img1, img2, kpts1, kpts2, conf = data
        # grd, sat, tgt, Rgt = data
        # B, _, sat_size, _ = sat.shape

        
        # Normalize ground truth translation
        # tgt = (tgt / sat_size) * grid_size_h

        # Move data to device
        img1, img2, kpts1, kpts2, conf = img1.to(device), img2.to(device), kpts1.to(device), kpts2.to(device), conf.to(device)

        # Forward pass through the feature extractor
        img1_feature, img2_feature = shared_feature_extractor(img1), shared_feature_extractor(img2)

        # Obtain matching scores and descriptors
        CVM_model.train()
        matching_score, img2_desc, img1_desc, img2_indices_topk, img1_indices_topk, matching_score_original = CVM_model(img1_feature, img2_feature)
        # kpts1_pred, kpts2_pred = CVM_model(img1_feature, img2_feature)

        print("matching_score.shape:", matching_score.shape)
        print("indices shape:", img1_indices_topk.shape, img2_indices_topk.shape)
        print("matching_score_original.shape:", matching_score_original.shape)
        print("img1_indices_topk:", img1_indices_topk)
        print("img2_indices_topk:", img2_indices_topk)
        # print("kpts1_pred.shape:", kpts1_pred.shape)
        # print("kpts2_pred.shape:", kpts2_pred.shape)
        print("matching_score:", matching_score)

        # Compute patch matches from gt keypoints
        patches_1, patches_2 = compute_patch_matches(kpts1.squeeze(0), kpts2.squeeze(0), img1.shape[2:], patch_size=14)
        print("patches_1 shape:", patches_1.shape)
        print("patches_2 shape:", patches_2.shape)
        print("patches_1:", patches_1)
        print("patches_1 max min:", patches_1.max(), patches_1.min())
        print("patches_2:", patches_2)
        print("patches_2 max min:", patches_2.max(), patches_2.min())
        
        # Compute losses
        # img1_predictions = torch.zeros((1, 256, 28*37))
        # img2_predictions = torch.zeros((1, 256, 28*37))
        # # Extract diagonal scores: matching_score[0, i, i] for i in 0..255
        # diag_scores = torch.diagonal(matching_score, dim1=1, dim2=2)  # shape: (1, 256)

        # # Scatter scores into logits
        # img1_predictions.scatter_(1, img1_indices_topk, diag_scores)
        # img2_predictions.scatter_(1, img2_indices_topk, diag_scores)
        print(matching_score.squeeze(0)[img1_indices_topk.squeeze(0), :].squeeze(0)[:patches_1.shape[0], :].shape, patches_1.shape)
        print(matching_score.squeeze(0).transpose(1, 0)[img2_indices_topk.squeeze(0), :].squeeze(0)[:patches_2.shape[0], :].shape, patches_2.shape)
        distance_loss = F.cross_entropy(matching_score.squeeze(0)[img1_indices_topk.squeeze(0), :].squeeze()[:patches_1.shape[0], :], patches_1.to(device)) + F.cross_entropy(matching_score.squeeze(0).transpose(1, 0)[img2_indices_topk.squeeze(0), :].squeeze()[:patches_2.shape[0], :], patches_2.to(device))
        print("distance_loss:", distance_loss.item())
        # Backpropagation
        optimizer.zero_grad()
        print("requires_grad:", distance_loss.requires_grad, matching_score.requires_grad)  # should be True
        distance_loss.backward()
        optimizer.step()

        # Log training loss
        writer.add_scalar("loss/distance", distance_loss.item(), global_step)
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
    if epoch % 10 == 0:
        print(f'📊 Epoch {epoch} - Evaluating on training set...')
        results_dir = '../results/'+label+'/'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        with torch.no_grad():
            CVM_model.eval()
            translation_error, yaw_error = [], []

            # for i, data in enumerate(train_dataloader, 0):
            data = train_dataloader.dataset[torch.randint(len(train_dataloader.dataset))]
            grd, sat, kpts1, kpts2 = data
            B, _, sat_size, _ = sat.shape

            grd, sat, kpts1, kpts2 = grd.to(device), sat.to(device), kpts1.to(device), kpts2.to(device)

            grd_feature, sat_feature = shared_feature_extractor(grd), shared_feature_extractor(sat)
    
            matching_score, sat_desc, grd_desc, sat_indices_topk, grd_indices_topk, matching_score_original = CVM_model(grd_feature, sat_feature)
            # _, num_kpts_sat, num_kpts_grd = matching_score.shape
            patches_1, patches_2 = compute_patch_matches(kpts1.squeeze(0), kpts2.squeeze(0), img1.shape[2:], patch_size=14)
            visualize_patch_matches(grd.squeeze(0).permute(1,2,0).cpu().numpy(), sat.squeeze(0).permute(1,2,0).cpu().numpy(), list(zip(patches_1.cpu().numpy(), patches_2.cpu().numpy())), patch_size=14)
                # # Sample training matches
                # matches_row = matching_score.flatten(1)
                # batch_idx = torch.arange(B).view(B, 1).repeat(1, num_samples_matches).reshape(B, num_samples_matches)
                # sampled_matching_idx = torch.multinomial(matches_row, num_samples_matches)

                # sampled_matching_row = torch.div(sampled_matching_idx, num_kpts_grd, rounding_mode='trunc')
                # sampled_matching_col = sampled_matching_idx % num_kpts_grd

                # sat_indices_sampled = torch.gather(sat_indices_topk.squeeze(1), 1, sampled_matching_row)
                # grd_indices_sampled = torch.gather(grd_indices_topk.squeeze(1), 1, sampled_matching_col)

                # X, Y, weights = metric_coord_sat_B[batch_idx, sat_indices_sampled, :], metric_coord_grd_B[batch_idx, grd_indices_sampled, :], matches_row[batch_idx, sampled_matching_idx]
                # R, t, ok_rank = weighted_procrustes_2d(X, Y, use_weights=True, use_mask=True, w=weights)

                # if t is None:
                #     print('⚠️ Skipping batch: Singular transformation matrix')
                #     continue
                
                # # Compute translation error
                # t = (t / grid_size_h) * sat_size
                # translation_error.extend(torch.norm(t - tgt, dim=-1).cpu().numpy())

                # # Compute yaw error
                # Rgt_np, R_np = Rgt.cpu().numpy(), R.cpu().numpy()
                # for b in range(B):
                #     cos = R_np[b,0,0]
                #     sin = R_np[b,1,0]
                #     yaw = np.degrees( np.arctan2(sin, cos) )            
                    
                #     cos_gt = Rgt_np[b,0,0]
                #     sin_gt = Rgt_np[b,1,0]
                    
                #     yaw_gt = np.degrees( np.arctan2(sin_gt, cos_gt) )
                    
                #     diff = np.abs(yaw - yaw_gt)
        
                    # yaw_error.append(np.min([diff, 360-diff]))          

            # translation_error_mean = np.mean(translation_error)    
            # translation_error_median = np.median(translation_error)
            
            # yaw_error_mean = np.mean(yaw_error)    
            # yaw_error_median = np.median(yaw_error) 
        
            # print(f'📉 Mean Translation Error: {translation_error_mean:.3f}')
            # print(f'📉 Median Translation Error: {translation_error_median:.3f}')
            # print(f'📉 Mean Yaw Error: {yaw_error_mean:.3f}')
            # print(f'📉 Median Yaw Error: {yaw_error_median:.3f}')
            
            # file = results_dir+'Mean_distance_error.txt'
            # with open(file,'ab') as f:
            #     np.savetxt(f, [translation_error_mean], fmt='%4f', header='Training_set_mean_distance_error_in_pixels:', comments=str(epoch)+'_')
        
            # file = results_dir+'Median_distance_error.txt'
            # with open(file,'ab') as f:
            #     np.savetxt(f, [translation_error_median], fmt='%4f', header='Training_set_median_distance_error_in_pixels:', comments=str(epoch)+'_')
        
            # file = results_dir+'Mean_orientation_error.txt'
            # with open(file,'ab') as f:
            #     np.savetxt(f, [yaw_error_mean], fmt='%4f', header='Training_set_mean_yaw_error:', comments=str(epoch)+'_')
        
            # file = results_dir+'Median_orientation_error.txt'
            # with open(file,'ab') as f:
            #     np.savetxt(f, [yaw_error_median], fmt='%4f', header='Training_set_median_yaw_error:', comments=str(epoch)+'_')

    # print(f'📊 Epoch {epoch} - Evaluating on validation set...')
    # with torch.no_grad():
    #     CVM_model.eval()
    #     translation_error, yaw_error = [], []

    #     for i, data in enumerate(train_dataloader, 0):
    #         grd, sat, tgt, Rgt = data
    #         B, _, sat_size, _ = sat.shape

    #         grd, sat, tgt, Rgt = grd.to(device), sat.to(device), tgt.to(device), Rgt.to(device)

    #         grd_feature, sat_feature = shared_feature_extractor(grd), shared_feature_extractor(sat)
    
    #         matching_score, sat_desc, grd_desc, sat_indices_topk, grd_indices_topk, matching_score_original = CVM_model(grd_feature, sat_feature)
    #         _, num_kpts_sat, num_kpts_grd = matching_score.shape

    #         # Sample validation matches
    #         matches_row = matching_score.flatten(1)
    #         batch_idx = torch.arange(B).view(B, 1).repeat(1, num_samples_matches).reshape(B, num_samples_matches)
    #         sampled_matching_idx = torch.multinomial(matches_row, num_samples_matches)

    #         sampled_matching_row = torch.div(sampled_matching_idx, num_kpts_grd, rounding_mode='trunc')
    #         sampled_matching_col = sampled_matching_idx % num_kpts_grd

    #         sat_indices_sampled = torch.gather(sat_indices_topk.squeeze(1), 1, sampled_matching_row)
    #         grd_indices_sampled = torch.gather(grd_indices_topk.squeeze(1), 1, sampled_matching_col)

    #         X, Y, weights = metric_coord_sat_B[batch_idx, sat_indices_sampled, :], metric_coord_grd_B[batch_idx, grd_indices_sampled, :], matches_row[batch_idx, sampled_matching_idx]
    #         R, t, ok_rank = weighted_procrustes_2d(X, Y, use_weights=True, use_mask=True, w=weights)

    #         if t is None:
    #             print('⚠️ Skipping batch: Singular transformation matrix')
    #             continue
            
    #         # Compute translation error
    #         t = (t / grid_size_h) * sat_size
    #         translation_error.extend(torch.norm(t - tgt, dim=-1).cpu().numpy())

    #         # Compute yaw error
    #         Rgt_np, R_np = Rgt.cpu().numpy(), R.cpu().numpy()
    #         for b in range(B):
    #             cos = R_np[b,0,0]
    #             sin = R_np[b,1,0]
    #             yaw = np.degrees( np.arctan2(sin, cos) )            
                
    #             cos_gt = Rgt_np[b,0,0]
    #             sin_gt = Rgt_np[b,1,0]
                
    #             yaw_gt = np.degrees( np.arctan2(sin_gt, cos_gt) )
                
    #             diff = np.abs(yaw - yaw_gt)
    
    #             yaw_error.append(np.min([diff, 360-diff]))          

    #     translation_error_mean = np.mean(translation_error)    
    #     translation_error_median = np.median(translation_error)
         
    #     yaw_error_mean = np.mean(yaw_error)    
    #     yaw_error_median = np.median(yaw_error) 
    
    #     print(f'📉 Mean Translation Error: {translation_error_mean:.3f}')
    #     print(f'📉 Median Translation Error: {translation_error_median:.3f}')
    #     print(f'📉 Mean Yaw Error: {yaw_error_mean:.3f}')
    #     print(f'📉 Median Yaw Error: {yaw_error_median:.3f}')
        
    #     file = results_dir+'Mean_distance_error.txt'
    #     with open(file,'ab') as f:
    #         np.savetxt(f, [translation_error_mean], fmt='%4f', header='Validation_set_mean_distance_error_in_pixels:', comments=str(epoch)+'_')
    
    #     file = results_dir+'Median_distance_error.txt'
    #     with open(file,'ab') as f:
    #         np.savetxt(f, [translation_error_median], fmt='%4f', header='Validation_set_median_distance_error_in_pixels:', comments=str(epoch)+'_')
    
    #     file = results_dir+'Mean_orientation_error.txt'
    #     with open(file,'ab') as f:
    #         np.savetxt(f, [yaw_error_mean], fmt='%4f', header='Validation_set_mean_yaw_error:', comments=str(epoch)+'_')
    
    #     file = results_dir+'Median_orientation_error.txt'
    #     with open(file,'ab') as f:
    #         np.savetxt(f, [yaw_error_median], fmt='%4f', header='Validation_set_median_yaw_error:', comments=str(epoch)+'_')
            
        

writer.flush()
