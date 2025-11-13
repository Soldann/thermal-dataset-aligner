import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import desc_l2norm
from torch.hub import load_state_dict_from_url
from DINO_modules.dinov2 import vit_large
from att_layers.transformer import Transformer_self_att
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention

import matplotlib.pyplot as plt

import configparser
import ast
config = configparser.ConfigParser()
config.read("./config.ini")
dataset = config["Dataset"]["dataset"]
if dataset == 'RobotCar':
    raw_ground_image_size = ast.literal_eval(config.get("RobotCar", "raw_ground_image_size"))
    cropped_ground_image_size = ast.literal_eval(config.get("RobotCar", "cropped_ground_image_size"))
    ground_image_size = ast.literal_eval(config.get("RobotCar", "ground_image_size"))
if dataset == 'RGBT_Scenes':
    raw_ground_image_size = ast.literal_eval(config.get("RGBT_Scenes", "raw_ground_image_size"))
    cropped_ground_image_size = ast.literal_eval(config.get("RGBT_Scenes", "cropped_ground_image_size"))
    ground_image_size = ast.literal_eval(config.get("RGBT_Scenes", "ground_image_size"))

class DinoExtractor(nn.Module):
    """
    DINOv2 Feature Extractor using a ViT-L/14 backbone.
    """

    def __init__(self, dinov2_weights=None):
        super().__init__()

        # Define DINOv2 extractor parameters
        self.dino_channels = 1024
        self.dino_downfactor = 14
        self.amp_dtype = torch.float16  # Define float precision

        if dinov2_weights is None:
            dinov2_weights = load_state_dict_from_url(
                "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
                map_location="cpu"
            )

        vit_kwargs = dict(
            img_size=518,
            patch_size=14,
            init_values=1.0,
            ffn_layer="mlp",
            block_chunks=0,
        )

        self.dinov2_vitl14 = vit_large(**vit_kwargs)
        self.dinov2_vitl14.load_state_dict(dinov2_weights)
        self.dinov2_vitl14.requires_grad_(False)
        self.dinov2_vitl14.eval()
        self.dinov2_vitl14.to(self.amp_dtype)

    def forward(self, x):
        B, C, H, W = x.shape
        # Ensure spatial dimensions are divisible by dino_downfactor
        x = x[:, :, : self.dino_downfactor * (H // self.dino_downfactor),
                 : self.dino_downfactor * (W // self.dino_downfactor)]

        with torch.no_grad():
            features = self.dinov2_vitl14.forward_features(x.to(self.amp_dtype))
            features = features["x_norm_patchtokens"].permute(0, 2, 1).reshape(
                B, self.dino_channels, H // self.dino_downfactor, W // self.dino_downfactor
            ).float()

        return features


class BasicBlock(nn.Module):
    """Pre-activation version of the BasicBlock (ResNet)."""
    
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=True, padding_mode='zeros'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, padding_mode=padding_mode)
        self.bn1 = nn.BatchNorm2d(planes) if bn else nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, padding_mode=padding_mode)
        self.bn2 = nn.BatchNorm2d(planes) if bn else nn.Identity()

        self.shortcut = (
            nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            if stride != 1 or in_planes != self.expansion * planes
            else nn.Identity()
        )

    def forward(self, x, relu=True):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out)) + self.shortcut(x)
        return F.relu(out) if relu else out


class DeepResBlockDesc(nn.Module):
    """
    Deep Residual Block for Descriptor Extraction.
    """

    def __init__(self, bn, last_dim, in_channels, block_dims, add_posEnc, norm_desc, padding_mode='zeros'):
        super().__init__()
        self.norm_desc = norm_desc

        self.resblocks = nn.Sequential(
            BasicBlock(in_channels, block_dims[0], bn=bn, padding_mode=padding_mode),
            BasicBlock(block_dims[0], block_dims[1], bn=bn, padding_mode=padding_mode),
            BasicBlock(block_dims[1], block_dims[2], bn=bn, padding_mode=padding_mode),
        )

        self.att_layer = Transformer_self_att(d_model=128, num_layers=3, add_posEnc=add_posEnc)
        self.final_block = BasicBlock(block_dims[2], last_dim, bn=bn, padding_mode=padding_mode)

    def forward(self, x):
        x = self.resblocks(x)
        x = self.att_layer(x)
        x = self.final_block(x, relu=False)

        return desc_l2norm(x) if self.norm_desc else x


class DeepResBlockDet(nn.Module):
    """
    Deep Residual Block for Keypoint Detection.
    """

    def __init__(self, bn, in_channels, block_dims, add_posEnc, use_softmax, padding_mode='zeros'):
        super().__init__()

        self.resblocks = nn.Sequential(
            BasicBlock(in_channels, block_dims[0], bn=bn, padding_mode=padding_mode),
            BasicBlock(block_dims[0], block_dims[1], bn=bn, padding_mode=padding_mode),
            BasicBlock(block_dims[1], block_dims[2], bn=bn, padding_mode=padding_mode),
        )

        self.att_layer = Transformer_self_att(d_model=128, num_layers=3, add_posEnc=add_posEnc)
        self.final_block = BasicBlock(block_dims[2], block_dims[3], bn=bn, padding_mode=padding_mode)
        self.score = nn.Conv2d(block_dims[3], 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.use_softmax = use_softmax
        self.eps = 1e-16

    def remove_borders(self, score_map, borders=3):
        """Removes detections near the borders."""
        mask = torch.ones_like(score_map)
        mask[:, :, :borders, :] = 0
        mask[:, :, :, :borders] = 0
        mask[:, :, -borders:, :] = 0
        mask[:, :, :, -borders:] = 0
        return mask * score_map

    def forward(self, x):
        x = self.resblocks(x)
        x = self.att_layer(x)
        x = self.final_block(x)

        scores = self.score(x)
        # plt.imshow(scores[0].permute(1,2,0).cpu().detach().numpy())
        # plt.colorbar()
        # plt.show()

        
        return (
            self.remove_borders(F.softmax(scores, dim=-1), 3) if self.use_softmax
            else self.remove_borders(torch.sigmoid(scores), 3)
        )

# class DeepResBlock_depth(torch.nn.Module):
#     def __init__(self, bn, in_channels, block_dims, add_posEnc, use_depth_sigmoid, max_depth, padding_mode = 'zeros'):
#         super().__init__()

#         self.use_depth_sigmoid = use_depth_sigmoid
#         self.max_depth = max_depth
#         self.sigmoid = torch.nn.Sigmoid()

#         self.resblock1 = BasicBlock(in_channels, block_dims[0], stride=1, bn=bn, padding_mode=padding_mode)
#         self.resblock2 = BasicBlock(block_dims[0], block_dims[1], stride=1, bn=bn, padding_mode=padding_mode)
#         self.resblock3 = BasicBlock(block_dims[1], block_dims[2], stride=1, bn=bn, padding_mode=padding_mode)
#         self.resblock4 = BasicBlock(block_dims[2], block_dims[3], stride=1, bn=bn, padding_mode=padding_mode)

#         self.depth = nn.Conv2d(block_dims[3], 1, kernel_size=1, stride=1, padding=0, bias=False)

#         self.att_layer = Transformer_self_att(d_model=128, num_layers=3, add_posEnc=add_posEnc)

#     def forward(self, feature_volume):

#         x = self.resblock1(feature_volume)
#         x = self.resblock2(x)
#         x = self.resblock3(x)
#         x = self.att_layer(x)
#         x = self.resblock4(x)

#         # Predict xy depths
#         # depths = torch.clip(self.depth(x), min=1e-3, max=500)
#         if self.use_depth_sigmoid:
#             depths = self.max_depth * self.sigmoid(self.depth(x))
#         else:
#             depths = self.depth(x)

#         return depths

class DeepResBlockDet_with_mask(nn.Module):
    """
    Deep Residual Block for Keypoint Detection.
    """

    def __init__(self, bn, in_channels, block_dims, add_posEnc, use_softmax, padding_mode='zeros'):
        super().__init__()

        self.resblocks = nn.Sequential(
            BasicBlock(in_channels, block_dims[0], bn=bn, padding_mode=padding_mode),
            BasicBlock(block_dims[0], block_dims[1], bn=bn, padding_mode=padding_mode),
            BasicBlock(block_dims[1], block_dims[2], bn=bn, padding_mode=padding_mode),
        )

        self.att_layer = Transformer_self_att(d_model=128, num_layers=3, add_posEnc=add_posEnc)
        self.final_block = BasicBlock(block_dims[2], block_dims[3], bn=bn, padding_mode=padding_mode)
        self.score = nn.Conv2d(block_dims[3], 1, kernel_size=1, bias=False)

        self.use_softmax = use_softmax
        self.eps = 1e-16

    def remove_borders(self, score_map, borders=3):
        """Removes detections near the borders."""
        mask = torch.ones_like(score_map)
        mask[:, :, :borders, :] = 0
        mask[:, :, :, :borders] = 0
        mask[:, :, -borders:, :] = 0
        mask[:, :, :, -borders:] = 0
        return mask * score_map

    def forward(self, x, keep_index):
        
        x = self.resblocks(x)
        x = self.att_layer(x)
        x = self.final_block(x)

        scores = self.score(x)

        scores_dims = scores.shape

        if self.use_softmax:
            scores = self.remove_borders(F.softmax(scores / 100, dim=-1), 3)
        else:
            self.remove_borders(torch.sigmoid(scores), 3)

        keep_index = keep_index.flatten(0)
        scores = scores.flatten(2)
        for b in range(scores_dims[0]):
            scores[b, :, ~keep_index] = 0

        scores = scores.view(scores_dims)
        
        return scores
        
class SelfAttention(nn.Module):
    """
    Self-Attention Mechanism for Feature Refinement.
    """

    def __init__(self, device, bev_res, embed_dim=128):
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.bev_res = bev_res

        # Generate BEV grid reference points
        grd_row, grd_col = np.meshgrid(np.linspace(0, 1, bev_res), np.linspace(0, 1, bev_res), indexing='ij')
        self.grd_reference_points = torch.tensor(np.stack((grd_col, grd_row), axis=-1), dtype=torch.float32).to(device)
        self.grd_reference_points = self.grd_reference_points.view(-1, 2).unsqueeze(1)

        # Spatial Shape & Level Start Index
        self.grd_spatial_shape = torch.tensor([[bev_res, bev_res]], device=device, dtype=torch.long)
        self.level_start_index = torch.tensor([0], device=device, dtype=torch.long)

        # Self-Attention Mechanism
        self.grd_attention_self = MultiScaleDeformableAttention(embed_dims=embed_dim, num_heads=8, num_levels=1, num_points=4, batch_first=True)

    def forward(self, query):
        bs = query.size(0)
        residual = query
        grd_reference_points = self.grd_reference_points.unsqueeze(0).expand(bs, -1, -1, -1)

        # Apply Self-Attention
        grd_bev = self.grd_attention_self(query=query, value=query, reference_points=grd_reference_points,
                                          spatial_shapes=self.grd_spatial_shape, level_start_index=self.level_start_index)

        return grd_bev + residual

class SelfAttention_robotcar(nn.Module):
    """
    Self-Attention Mechanism for Feature Refinement.
    """

    def __init__(self, device, bev_res, bev_height_res=None, embed_dim=128):
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.bev_res = bev_res
        self.bev_height_res = bev_height_res if bev_height_res is not None else int(np.floor(bev_res/2))+1

        # Generate BEV grid reference points
        grd_row, grd_col = np.meshgrid(np.linspace(0, 1, int(np.floor(bev_res/2))+1), np.linspace(0, 1, bev_res), indexing='ij')
        # grd_row, grd_col = np.meshgrid(np.linspace(0, 1, self.bev_height_res), np.linspace(0, 1, bev_res), indexing='ij')
        self.grd_reference_points = torch.tensor(np.stack((grd_col, grd_row), axis=-1), dtype=torch.float32).to(device)
        self.grd_reference_points = self.grd_reference_points.view(-1, 2).unsqueeze(1)

        # Spatial Shape & Level Start Index
        self.grd_spatial_shape = torch.tensor([[int(np.floor(bev_res/2))+1, bev_res]], device=device, dtype=torch.long)
        # self.grd_spatial_shape = torch.tensor([[self.bev_height_res, bev_res]], device=device, dtype=torch.long)
        self.level_start_index = torch.tensor([0], device=device, dtype=torch.long)
        # Self-Attention Mechanism
        self.grd_attention_self = MultiScaleDeformableAttention(embed_dims=embed_dim, num_heads=8, num_levels=1, num_points=4, batch_first=True)

    def forward(self, query):
        bs = query.size(0)
        residual = query
        grd_reference_points = self.grd_reference_points.unsqueeze(0).expand(bs, -1, -1, -1)

        # Apply Self-Attention
        grd_bev = self.grd_attention_self(query=query, value=query, reference_points=grd_reference_points,
                                          spatial_shapes=self.grd_spatial_shape, level_start_index=self.level_start_index)
        return grd_bev + residual
        
class CrossAttention(nn.Module):
    """
    Cross-Attention Mechanism for BEV to Image Feature Mapping.
    """

    def __init__(self, device, bev_res, height_res, embed_dim=128, grid_size_h=71, grid_size_v=50):
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.bev_res = bev_res
        self.height_res = height_res
        
        eps = 1e-6  # Prevent division by zero

        # Define 3D voxel grid in point cloud space
        x = np.linspace(-grid_size_h / 2, grid_size_h / 2, bev_res)
        y = np.linspace(-grid_size_h / 2, grid_size_h / 2, bev_res)
        z = np.linspace(-grid_size_v / 2, grid_size_v / 2, height_res)
        grd_x, grd_y, grd_z = np.meshgrid(x, y, z, indexing='ij')

        # Convert to spherical coordinates for reference sampling
        phi = np.sign(grd_y + eps) * np.arccos(grd_x / (np.sqrt(grd_x**2 + grd_y**2) + eps))
        theta = np.arccos(grd_z / (np.sqrt(grd_x**2 + grd_y**2 + grd_z**2) + eps))

        # Adjust phi to align ground panorama orientation
        phi = 2 * np.pi - phi
        phi[phi > 2 * np.pi] -= 2 * np.pi

        # Convert to normalized coordinates
        self.grd_col_cross = torch.tensor(phi / (2 * np.pi), dtype=torch.float32, device=device)
        self.grd_row_cross = torch.tensor(theta / np.pi, dtype=torch.float32, device=device)

        # Define Spatial Shape & Level Start Index
        self.grd_spatial_shape_cross = torch.tensor([[51, 102]], device=device, dtype=torch.long)
        self.level_start_index = torch.tensor([0], device=device, dtype=torch.long)

        # Cross-Attention Mechanism
        self.grd_attention_cross = MultiScaleDeformableAttention(embed_dims=embed_dim, num_heads=8, num_levels=1, num_points=4, batch_first=True)

        # Projection layer for attention weighting
        self.projector = nn.Linear(embed_dim, 1)

    def forward(self, query, value):
        bs = query.size(0)
        residual = query

        grd_bev_list = []
        for i in range(self.height_res):
            # Prepare reference points for cross-attention
            grd_reference_points = torch.stack((self.grd_col_cross[:, :, i], self.grd_row_cross[:, :, i]), dim=-1)
            grd_reference_points = grd_reference_points.view(-1, 2).unsqueeze(1).unsqueeze(0).expand(bs, -1, -1, -1)

            # Apply Cross-Attention
            grd_bev = self.grd_attention_cross(query=query, value=value, reference_points=grd_reference_points,
                                               spatial_shapes=self.grd_spatial_shape_cross, level_start_index=self.level_start_index)
            
            grd_bev_list.append(grd_bev.view(bs, self.bev_res, self.bev_res, self.embed_dim))

        # Stack results into a 4D tensor: [B, bev_res, bev_res, embed_dim, height_res]
        grd_3d = torch.stack(grd_bev_list, dim=-1).permute(0, 1, 2, 4, 3)

        # Compute soft attention weights over height dimension
        weights = F.softmax(self.projector(grd_3d), dim=3)
        max_height_index = torch.argmax(weights, dim=3)

        # Compute weighted sum over height dimension
        grd_bev = (weights * grd_3d).sum(dim=3)

        return grd_bev.view(bs, -1, self.embed_dim) + residual, max_height_index

class CrossAttention_robotcar(nn.Module):
    """
    Cross-Attention Mechanism for BEV to Image Feature Mapping.
    """

    def __init__(self, device, bev_res, height_res, embed_dim=128, grid_size_h=71, grid_size_v=50):
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.bev_res = bev_res
        self.height_res = height_res

        eps = 1e-6  # Prevent division by zero

        # Define 3D voxel grid in point cloud space
        x = np.linspace(-grid_size_h / 2, 0, int(np.floor(bev_res/2))+1)
        y = np.linspace(-grid_size_h / 2, grid_size_h / 2, bev_res)
        z = np.linspace(-grid_size_v / 2, grid_size_v / 2, height_res)
        
        grd_x, grd_y, grd_z = np.meshgrid(x, y, z, indexing='ij')

        # Convert coordinates for reference sampling
        fx, fy = 964.828979 * ground_image_size[1] / raw_ground_image_size[1], 964.828979 * ground_image_size[0] / raw_ground_image_size[0] 
        # fx, fy = 964.828979, 964.828979  
        cx, cy = (643.788025 - (raw_ground_image_size[1] - cropped_ground_image_size[1]) / 2) * ground_image_size[1] / cropped_ground_image_size[1], 484.407990 * ground_image_size[0] / raw_ground_image_size[0]  
        # cx, cy = 643.788025 , 484.407990   
        camera_k = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        
        # self.grid_3d = torch.stack([torch.tensor(-grd_x), torch.tensor(grd_y), torch.tensor(-grd_z)], dim=-1) # to align camera coordinate system, x pointing outwards, y pointing to the right, z pointing down
        self.grid_3d = torch.stack([torch.tensor(grd_y), torch.tensor(-grd_z), torch.tensor(-grd_x)], dim=-1).reshape(-1, 3).to(torch.float).to(device) # to align camera coordinate system, z pointing outwards, y pointing down, x pointing to the right
            
        self.grid_3d = self.grid_3d.reshape(-1, 3).to(torch.float).to(device) 
        grid_3d = self.grid_3d.T
        
        projected_points = torch.tensor(camera_k).to(self.device) @ grid_3d
        
        self.u = (projected_points[0, :] / projected_points[2, :] / ground_image_size[1]).view(int(np.floor(bev_res/2))+1, bev_res, height_res)
        self.v = (projected_points[1, :] / projected_points[2, :] / ground_image_size[0]).view(int(np.floor(bev_res/2))+1, bev_res, height_res)
        
        self.keep_index = torch.logical_and(torch.logical_and(self.u >= 0, self.u <= 1), torch.logical_and(self.v >= 0, self.v <= 1)).sum(-1).to(torch.bool)
                
        # Define Spatial Shape & Level Start Index
        self.grd_spatial_shape_cross = torch.tensor([[ground_image_size[0]/14, ground_image_size[1]/14]], device=device, dtype=torch.long)
        self.level_start_index = torch.tensor([0], device=device, dtype=torch.long)

        # Cross-Attention Mechanism
        self.grd_attention_cross = MultiScaleDeformableAttention(embed_dims=embed_dim, num_heads=8, num_levels=1, num_points=4, batch_first=True)

        # Projection layer for attention weighting
        self.projector = nn.Linear(embed_dim, 1)

    def forward(self, query, value):
        bs = query.size(0)
        residual = query

        grd_bev_list = []
        for i in range(self.height_res):
            # Prepare reference points for cross-attention
            grd_reference_points = torch.stack((self.u[:,:,i], self.v[:,:,i]), dim=-1)
            grd_reference_points = grd_reference_points.view(-1, 2).unsqueeze(1).unsqueeze(0).expand(bs, -1, -1, -1)
            
            # Apply Cross-Attention
            grd_bev = self.grd_attention_cross(query=query, value=value, reference_points=grd_reference_points,
                                               spatial_shapes=self.grd_spatial_shape_cross, level_start_index=self.level_start_index)
            
            grd_bev_list.append(grd_bev.view(bs, int(np.floor(self.bev_res/2))+1, self.bev_res, self.embed_dim))

        # Stack results into a 4D tensor: [B, bev_res / 2 + 1, bev_res, embed_dim, height_res]
        grd_3d = torch.stack(grd_bev_list, dim=-1).permute(0, 1, 2, 4, 3)       

        # Compute soft attention weights over height dimension
        weights = F.softmax(self.projector(grd_3d), dim=3)
        max_height_index = torch.argmax(weights, dim=3)

        # Compute weighted sum over height dimension
        grd_bev = (weights * grd_3d).sum(dim=3)

        return grd_bev.view(bs, -1, self.embed_dim) + residual, max_height_index


class ScalePredictor(nn.Module):
    def __init__(self, img_feat_dim):
        super().__init__()

        # Depth encoder: encodes the downsampled depth map into compact feature
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Global pooling for invariance
        )

        # Fully connected layers to predict scale
        self.fc = nn.Sequential(
            nn.Linear(img_feat_dim + 32, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)  # Output: [scale_raw]
        )

    def forward(self, img_features, depth_map):
        """
        Args:
            img_features: [B, C, H, W] - DINOv2 features (before pooling)
            depth_map: [B, 1, H, W] - downsampled relative depth map

        Returns:
            scale: [B,] - positive scale
        """
        # Encode depth features
        depth_features = self.depth_encoder(depth_map)  # [B, 32, 1, 1]
        depth_features = depth_features.view(depth_features.size(0), -1)  # [B, 32]

        # Global average pool image features (make invariant to panorama rotation)
        img_features_pooled = img_features.mean(dim=-1).mean(dim=-1)  # [B, img_feat_dim]

        # Fuse features
        fused_features = torch.cat([img_features_pooled, depth_features], dim=-1)  # [B, img_feat_dim + 32]

        # Predict scale 
        out = self.fc(fused_features)
        scale_raw = out[:, 0]

        # Force scale to be positive
        scale = torch.exp(torch.tanh(scale_raw))

        return scale