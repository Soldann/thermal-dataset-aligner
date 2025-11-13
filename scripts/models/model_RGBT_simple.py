import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import DeepResBlockDesc


import configparser
config = configparser.ConfigParser()
config.read("./config.ini")
sat_bev_res = config.getint("Model", "sat_bev_res")
eps = config.getfloat("Constants", "epsilon")




class CVM_Thermal_Simple(nn.Module):
    def __init__(self, device, temperature=0.1, num_keypoints=256,
                 embed_dim=1024, desc_dim=128):
        super(CVM_Thermal_Simple, self).__init__()
        self.device = device
        self.temperature = temperature
        self.embed_dim = embed_dim
        self.num_keypoints = num_keypoints

        bn = True
        block_dims = [512, 256, 128, 64]
        add_posEnc_grd = True
        add_posEnc_sat = False
        norm_desc = True
        
        self.dustbin_score = nn.Parameter(torch.tensor(1.))

        ### grd
        self.grd_projector = DeepResBlockDesc(bn, last_dim=desc_dim, in_channels=embed_dim, block_dims=block_dims, 
                                              add_posEnc=add_posEnc_grd, norm_desc=norm_desc)
        
        ### sat
        self.sat_projector = DeepResBlockDesc(bn, last_dim=desc_dim, in_channels=embed_dim, block_dims=block_dims, 
                                              add_posEnc=add_posEnc_sat, norm_desc=norm_desc)
        
    
    def forward(self, grd_feature, sat_feature, mask=None):

        bs = grd_feature.size()[0]

        # Ground descriptors & scores
        grd_desc = self.grd_projector(grd_feature).flatten(2)

        # Interpolate satellite features & compute descriptors & scores
        # sat_feature_bev = F.interpolate(sat_feature, (sat_bev_res, sat_bev_res), mode='bilinear', align_corners=False)
        sat_desc = self.sat_projector(sat_feature).flatten(2)
        
        # Compute matching points
        matching_score_original = torch.matmul(sat_desc.transpose(1, 2).contiguous(), grd_desc) / self.temperature
        # matching_score_original = matching_score_original.masked_fill(
        #     ~mask.unsqueeze(1), float('-inf')
        # )

        # Construct Matching Score Matrix with Dustbin
        b, m, n = matching_score_original.shape
        bins0 = self.dustbin_score.expand(b, m, 1)
        bins1 = self.dustbin_score.expand(b, 1, n)
        alpha = self.dustbin_score.expand(b, 1, 1)

        couplings = torch.cat([torch.cat([matching_score_original, bins0], -1),
                               torch.cat([bins1, alpha], -1)], 1)

        couplings = F.softmax(couplings, dim=1) * F.softmax(couplings, dim=2)
        matching_score = couplings[:, :-1, :-1]

        # Extract topk matches
        k = self.num_keypoints
        flat_scores = matching_score.reshape(b, -1)  # shape: (b, m * n)
        topk_values, topk_indices = torch.topk(flat_scores, k, dim=1)

        # Convert flat indices to 2D coordinates
        sat_indices_topk = topk_indices // n
        grd_indices_topk = topk_indices % n
        
        return matching_score_original, sat_indices_topk, grd_indices_topk, matching_score_original
        