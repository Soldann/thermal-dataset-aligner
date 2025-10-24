import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import (
    SelfAttention_robotcar as SelfAttention,
    CrossAttention_robotcar as CrossAttention,
    DeepResBlockDet,
    DeepResBlockDet_with_mask,
    DeepResBlockDesc
)
from mmcv.cnn.bricks.transformer import FFN
from utils.utils import matching_points


class CVM(nn.Module):
    """
    Cross-View Matching (CVM) Model for Ground-to-Satellite Feature Matching.

    Args:
        device (torch.device): Device on which to run the model.
        grd_bev_res (int): BEV resolution for the ground image.
        grd_height_res (int): Height resolution for the ground image.
        sat_bev_res (int): BEV resolution for the satellite image.
        temperature (float): Temperature scaling for similarity computation.
        num_keypoints (int): Number of top keypoints selected for matching.
        embed_dim (int): Dimension of the feature embeddings.
        desc_dim (int): Dimension of the descriptor output.
        grid_size_h (int): Horizontal grid size for feature extraction.
        grid_size_v (int): Vertical grid size for feature extraction.
    """

    def __init__(self, device, grd_bev_res, grd_height_res, sat_bev_res, temperature=0.1, 
                 num_keypoints=256, embed_dim=1024, desc_dim=128, grid_size_h=71, grid_size_v=50):
        super(CVM, self).__init__()
        self.device = device
        self.temperature = temperature
        self.num_keypoints = num_keypoints
        self.embed_dim = embed_dim
        self.grd_bev_res = grd_bev_res
        self.grd_height_res = grd_height_res
        self.sat_bev_res = sat_bev_res


        bn = True
        block_dims = [512, 256, 128, 64]
        norm_desc = True
        add_posEnc_grd = True
        add_posEnc_sat = False  # No positional encoding for satellite view

        self.dustbin_score = nn.Parameter(torch.tensor(1.0))

        # Ground Grid Queries
        self.grd_grid_queries = nn.Parameter(torch.rand(int(np.floor(grd_bev_res/2))+1, grd_bev_res, embed_dim), requires_grad=True)

        # Self-Attention & Cross-Attention Layers
        self.grd_attention_self = nn.ModuleList([SelfAttention(device, grd_bev_res, embed_dim) for _ in range(6)])
        self.grd_attention_cross = nn.ModuleList([CrossAttention(device, grd_bev_res, grd_height_res, embed_dim, grid_size_h=grid_size_h, grid_size_v=grid_size_v) for _ in range(6)])
        self.grd_ffn = nn.ModuleList([FFN(embed_dim) for _ in range(6)])

        # Layer Norms
        self.grd_layer_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(18)])

        # Ground Feature Processing
        self.grd_projector = DeepResBlockDesc(bn, last_dim=desc_dim, in_channels=embed_dim, block_dims=block_dims, 
                                              add_posEnc=add_posEnc_grd, norm_desc=norm_desc)
        self.grd_detector = DeepResBlockDet_with_mask(bn, in_channels=embed_dim, block_dims=block_dims, 
                                            add_posEnc=add_posEnc_grd, use_softmax=True)

        # Satellite Feature Processing
        self.sat_projector = DeepResBlockDesc(bn, last_dim=desc_dim, in_channels=embed_dim, block_dims=block_dims, 
                                              add_posEnc=add_posEnc_sat, norm_desc=norm_desc)
        self.sat_detector = DeepResBlockDet(bn, in_channels=embed_dim, block_dims=block_dims, 
                                            add_posEnc=add_posEnc_sat, use_softmax=True)

    def forward(self, grd_feature, sat_feature):
        """
        Forward pass for CVM model.

        Args:
            grd_feature (Tensor): Ground feature tensor of shape [B, C, H, W].
            sat_feature (Tensor): Satellite feature tensor of shape [B, C, H, W].

        Returns:
            matching_score (Tensor): Matching score matrix of shape [B, K, K].
            sat_desc (Tensor): Flattened satellite descriptors [B, D, N].
            grd_desc (Tensor): Flattened ground descriptors [B, D, N].
            sat_indices_topk (Tensor): Top-k satellite indices.
            grd_indices_topk (Tensor): Top-k ground indices.
        """
        bs = grd_feature.shape[0]

        # Initialize ground queries
        grd_query = self.grd_grid_queries.view(-1, self.embed_dim).unsqueeze(0).expand(bs, -1, -1)
        grd_value = grd_feature.permute(0, 2, 3, 1).reshape(bs, -1, self.embed_dim).contiguous()

        # Transformer Iterations (Self + Cross Attention + FFN)
        for i in range(6):
            idx = i * 3  # Index for layer norm groups

            grd_query = self.grd_attention_self[i](grd_query)
            grd_query = self.grd_layer_norms[idx](grd_query)

            grd_query, _ = self.grd_attention_cross[i](grd_query, grd_value)
            grd_query = self.grd_layer_norms[idx + 1](grd_query)

            grd_query = self.grd_ffn[i](grd_query)
            grd_query = self.grd_layer_norms[idx + 2](grd_query)

        # Reshape ground query for processing
        grd_query = grd_query.permute(0, 2, 1).reshape(bs, self.embed_dim, int(np.floor(self.grd_bev_res/2))+1, self.grd_bev_res)

        # Ground descriptors & scores
        grd_desc = self.grd_projector(grd_query)
        grd_scrs = self.grd_detector(grd_query, self.grd_attention_cross[i].keep_index)

        # Interpolate satellite features & compute descriptors & scores
        sat_feature_bev = F.interpolate(sat_feature, (self.sat_bev_res, self.sat_bev_res), mode='bilinear', align_corners=False)
        sat_desc = self.sat_projector(sat_feature_bev)
        sat_scrs = self.sat_detector(sat_feature_bev)


        # Compute matching points
        matching_score_original, sat_indices_topk, grd_indices_topk = matching_points(
            grd_desc, grd_scrs, sat_desc, sat_scrs, k=self.num_keypoints, temperature=self.temperature
        )


        # Construct Matching Score Matrix with Dustbin
        b, m, n = matching_score_original.shape
        bins0 = self.dustbin_score.expand(b, m, 1)
        bins1 = self.dustbin_score.expand(b, 1, n)
        alpha = self.dustbin_score.expand(b, 1, 1)

        couplings = torch.cat([torch.cat([matching_score_original, bins0], -1),
                               torch.cat([bins1, alpha], -1)], 1)

        couplings = F.softmax(couplings, dim=1) * F.softmax(couplings, dim=2)
        matching_score = couplings[:, :-1, :-1]

        return matching_score, sat_desc.flatten(2), grd_desc.flatten(2), sat_indices_topk, grd_indices_topk, matching_score_original
