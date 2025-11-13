import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import DeepResBlockDet, DeepResBlockDesc
from utils.utils import matching_points

import configparser
config = configparser.ConfigParser()
config.read("./config.ini")
sat_bev_res = config.getint("Model", "sat_bev_res")

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

    def __init__(self, device, sat_bev_res, temperature=0.1, 
                 num_keypoints=256, embed_dim=1024, desc_dim=128):
        super(CVM, self).__init__()
        self.device = device
        self.temperature = temperature
        self.num_keypoints = num_keypoints
        self.embed_dim = embed_dim
        self.sat_bev_res = sat_bev_res

        bn = True
        block_dims = [512, 256, 128, 64]
        norm_desc = True
        add_posEnc_grd = True
        add_posEnc_sat = False  # No positional encoding for satellite view

        self.dustbin_score = nn.Parameter(torch.tensor(1.0))        

        # Ground Feature Processing
        self.grd_projector = DeepResBlockDesc(bn, last_dim=desc_dim, in_channels=embed_dim, block_dims=block_dims, 
                                              add_posEnc=add_posEnc_grd, norm_desc=norm_desc)
        self.grd_detector = DeepResBlockDet(bn, in_channels=embed_dim, block_dims=block_dims, 
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

        # Ground descriptors & scores
        grd_desc = self.grd_projector(grd_feature)
        grd_scrs = self.grd_detector(grd_feature)

        # Interpolate satellite features & compute descriptors & scores
        sat_feature_bev = F.interpolate(sat_feature, (sat_bev_res, sat_bev_res), mode='bilinear', align_corners=False)
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