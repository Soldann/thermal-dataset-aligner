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
parser.add_argument('--num_images_to_log', type=int, help='number of images to log', default=5)

args = vars(parser.parse_args())
machine = args['machine']
learning_rate = args['learning_rate']
batch_size = args['batch_size']
beta = args['beta']
epoch_to_resume = args['epoch_to_resume']
experiment_name = args['name']
num_images_to_log = args['num_images_to_log']

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
import wandb
import cv2
import ahrs
from torch.nn import functional as F

from models.model_RGBT import CVM_Thermal
from models.model_RGBT_simple import CVM_Thermal_Simple
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
elif machine == 'scitas':
    dataset_root = config["RGBT_Scenes"]["scitas_dataset_root"]

batch_size = config.getint("RGBT_Scenes", "batch_size")
learning_rate = config.getfloat("Training", "learning_rate")
beta = config.getfloat("Loss", "beta")
epoch_to_resume = epoch_to_resume if epoch_to_resume > 0 else config.getint("Training", "epoch_to_resume")

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
train_data, val_data = RGBT_Scenes_Dataset.build_test_train_dataloaders(dataset_root, training_ratio=args['training_ratio'], low_memory_mode=True, image_mode=RGBT_Scenes_Dataset.ImagePairMode.thermal_to_thermal)
print(len(train_data), 'training samples found.')
print(len(val_data), 'validation samples found.')

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=train_data.collate_fn)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=val_data.collate_fn)
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

# Setup TensorBoard / wandb logging
run = wandb.init(project="semester-project", sync_tensorboard=True, config=config, name=label)
writer_dir = f'../tensorboard/{label}/'
os.makedirs(writer_dir, exist_ok=True)
writer = SummaryWriter(log_dir=writer_dir)

# -------------------------
# Training Loop
# -------------------------
for epoch in range(epoch_to_resume + 1, 100 + 1):
    print(f'🚀 Epoch {epoch} - Training...')
    
    running_loss = 0.0
    CVM_model.train()

    for i, data in enumerate(train_dataloader, 0):
        img1, img2, conf, patches_1, patches_2, patches_1_len, patches_2_len, img1_pose, img2_inv_pose, img1_K, img2_K = data
        print("Batch", i, "image shapes:", img1.shape, img2.shape)
        # grd, sat, tgt, Rgt = data
        # B, _, sat_size, _ = sat.shape

        
        # Normalize ground truth translation
        # tgt = (tgt / sat_size) * grid_size_h

        # Move data to device
        img1, img2, conf, patches_1, patches_2, patches_1_len, patches_2_len = img1.to(device), img2.to(device), conf.to(device), patches_1.long().to(device), patches_2.long().to(device), patches_1_len.to(device), patches_2_len.to(device)

        # Forward pass through the feature extractor
        img1_feature, img2_feature = shared_feature_extractor(img1), shared_feature_extractor(img2)

        # Obtain matching scores and descriptors
        matching_score, img2_indices_topk, img1_indices_topk, matching_score_original = CVM_model(img1_feature, img2_feature)
        # kpts1_pred, kpts2_pred = CVM_model(img1_feature, img2_feature)

        print("matching_score.shape:", matching_score.shape)
        
        # print("indices shape:", img1_indices_topk.shape, img2_indices_topk.shape)
        # print("matching_score_original.shape:", matching_score_original.shape)
        # print("img1_indices_topk:", img1_indices_topk)
        # print("img2_indices_topk:", img2_indices_topk)
        # # print("kpts1_pred.shape:", kpts1_pred.shape)
        # # print("kpts2_pred.shape:", kpts2_pred.shape)
        # print("matching_score_original:", matching_score_original)
        # print("matching_score:", matching_score)

        # Compute patch matches from gt keypoints
        # patches_1, patches_2 = compute_patch_matches(kpts1.squeeze(0), kpts2.squeeze(0), img1.shape[2:], patch_size=14)
        # max_keypoints_for_comparison = min(num_keypoints, patches_1.shape[0], patches_2.shape[0])
        print("patches_1 shape:", patches_1.shape, patches_1.dtype)
        print("patches_2 shape:", patches_2.shape, patches_2.dtype)
        # print("patches_1:", patches_1)
        # print("patches_1 max min:", patches_1.max(), patches_1.min())
        # print("patches_2:", patches_2)
        # print("patches_2 max min:", patches_2.max(), patches_2.min())
        
        # Compute losses
        # img1_predictions = torch.zeros((1, 256, 28*37))
        # img2_predictions = torch.zeros((1, 256, 28*37))
        # # Extract diagonal scores: matching_score[0, i, i] for i in 0..255
        # diag_scores = torch.diagonal(matching_score, dim1=1, dim2=2)  # shape: (1, 256)

        # # Scatter scores into logits
        # img1_predictions.scatter_(1, img1_indices_topk, diag_scores)
        # img2_predictions.scatter_(1, img2_indices_topk, diag_scores)

        # create a mask from patches_1_len and patches_2_len
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

        # print(matching_score.squeeze(0)[patches_2.squeeze(0), :].shape)
        # print(matching_score.squeeze(0).transpose(1, 0)[patches_1.squeeze(0), :].shape)
        # print("img1_indices_topk.shape:", img1_indices_topk.shape)
        # print("img2_indices_topk.shape:", img2_indices_topk.shape)
        # print("matching_score_after_indexes shape:", matching_score.squeeze(0)[img1_indices_topk.squeeze(0), :][:max_keypoints_for_comparison, :].shape)
        # print("matching_score_after_indexes shape:", matching_score.squeeze(0).transpose(1, 0)[img2_indices_topk.squeeze(0), :][:max_keypoints_for_comparison, :].shape)
        # print("img1 matching score size vs gt patch matches size: ", matching_score.squeeze(0)[patches_2, :].squeeze(0)[:max_keypoints_for_comparison, :].shape, patches_1.shape)
        # print("img2 matching score size vs gt patch matches size: ",matching_score.squeeze(0).transpose(1, 0)[patches_1, :].squeeze(0)[:max_keypoints_for_comparison, :].shape, patches_2.shape)
        img1_loss = F.cross_entropy(
            scores_img1[max_keypoints_mask],
            patches_1[max_keypoints_mask]
        )
        # img1_topk_loss = F.cross_entropy(
        #     matching_score[max_keypoints_mask],
        #     patches_1[max_keypoints_mask]
        # )
        img2_loss = F.cross_entropy(
                scores_img2[max_keypoints_mask],
                patches_2[max_keypoints_mask]
        )
        # img2_topk_loss = F.cross_entropy(
            # matching_score.squeeze(0).transpose(1, 0)[img2_indices_topk.squeeze(0), :][:max_keypoints_for_comparison, :],
            # patches_2.squeeze(0)[:max_keypoints_for_comparison].to(device)
        # )
        loss = img1_loss + img2_loss #+ img1_topk_loss + img2_topk_loss
        print("loss:", loss.item())
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log training loss
        writer.add_scalar("loss/train_error", loss.item(), global_step)
        global_step += 1


    model_dir = '../checkpoints/'+label+'/' + str(epoch) + '/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print('save checkpoint at '+model_dir)
    torch.save((global_step, torch.get_rng_state(), CVM_model.cpu().state_dict()), model_dir+'model.pt') # saving model
    # CVM_model.load_state_dict(torch.load(model_dir+'model.pt'))
    CVM_model.cuda() # moving model to GPU for further training
    
    # -------------------------
    # Validation 
    # -------------------------
    if epoch % 1 == 0:
        print(f'📊 Epoch {epoch} - Evaluating on validation set...')
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
                
                if i == visualization_index:
                    vis_imgs = [
                        visualize_patch_matches(img1[item_to_pick].permute(1,2,0).cpu().numpy(), img2[item_to_pick].permute(1,2,0).cpu().numpy(), list(zip(img1_indices_topk[item_to_pick].reshape(num_keypoints,).cpu().numpy(), img2_indices_topk[item_to_pick].reshape(num_keypoints,).cpu().numpy())), patch_size=14, patches_to_draw=256, headless=True)
                        for item_to_pick in range(min(B, num_images_to_log))
                    ]
                    wandb.log({"validation/patch_matches": [wandb.Image(vis_img) for vis_img in vis_imgs]}, step=epoch)

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

                homogenous_row = torch.tensor([0,0,0,1], dtype=torch.float32).view(1,1,4).repeat(B,1,1).to(device)
                c1Tw = torch.cat([img1_pose, homogenous_row], dim=1) # world to cam1
                wTc2 = torch.cat([img2_inv_pose, homogenous_row], dim=1) # cam2 to world
                c1Tc2 = torch.bmm(c1Tw, wTc2) # cam2 to cam1

                rotation_errors = []
                translation_errors = []
                for batch_item in range(B):
                    kpts1 = convert_patches_to_keypoints(img1_indices_topk[batch_item].reshape(num_keypoints,), img1.shape[2:]).cpu().numpy()
                    kpts2 = convert_patches_to_keypoints(img2_indices_topk[batch_item].reshape(num_keypoints,), img2.shape[2:]).cpu().numpy()
                    E, mask = cv2.findEssentialMat(kpts1, kpts2, img1_K[batch_item].cpu().numpy(), method=cv2.RANSAC, prob=0.999, threshold=1.0)
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

                loss = img1_loss + img2_loss # + img1_topk_loss + img2_topk_loss
                val_error.append(loss.item())
                overall_rotation_errors.extend(rotation_errors)
                overall_translation_errors.extend(translation_errors)
                if i == visualization_index:
                    writer.add_scalar("loss/val_error", loss.item(), global_step)
                    writer.add_scalar("error/rotation_error", np.mean(rotation_errors), global_step)
                    writer.add_scalar("error/translation_error", np.mean(translation_errors), global_step)
            
            val_error_mean = np.mean(val_error)    
            val_error_median = np.median(val_error)
            rotation_error_mean = np.mean(overall_rotation_errors)
            rotation_error_median = np.median(overall_rotation_errors)
            translation_error_mean = np.mean(overall_translation_errors)
            translation_error_median = np.median(overall_translation_errors)
        
            print(f'📉 Mean Distance Error: {val_error_mean:.3f}')
            print(f'📉 Median Distance Error: {val_error_median:.3f}')
            print(f'Mean Rotation Error: {rotation_error_mean:.3f}')
            print(f'Median Rotation Error: {rotation_error_median:.3f}')
            print(f'Mean Translation Error: {translation_error_mean:.3f}')
            print(f'Median Translation Error: {translation_error_median:.3f}')

            writer.add_scalar("error/mean_val_error", val_error_mean, epoch)
            writer.add_scalar("error/median_val_error", val_error_median, epoch)
            writer.add_scalar("error/mean_rotation_error", rotation_error_mean, epoch)
            writer.add_scalar("error/median_rotation_error", rotation_error_median, epoch)
            writer.add_scalar("error/mean_translation_error", translation_error_mean, epoch)
            writer.add_scalar("error/median_translation_error", translation_error_median, epoch)

writer.flush()
