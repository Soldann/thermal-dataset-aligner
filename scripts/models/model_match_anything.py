from pathlib import Path
import kornia as K
import kornia.feature as KF
# from kornia_moons.viz import draw_LAF_matches
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from match_anything.utils.model_utils import load_config
from match_anything.third_party.ROMA.roma.matchanything_roma_model import MatchAnything_Model


class ModelMatchAnything(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Get default configurations
        config = load_config("matchanything_roma")

        self.matcher = MatchAnything_Model(config, True)
        ckpt_path = Path(__file__).parent.parent.parent / "MatchAnything/weights/matchanything_roma.ckpt"
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.matcher.load_state_dict(ckpt['state_dict'])

        self.matcher.eval()
    def forward(self, img1, img2):
        input_dict = {
            "image0_rgb": img1.unsqueeze(0), # add back batch dimension
            "image1_rgb": img2.unsqueeze(0),
        }

        with torch.inference_mode():
            batch = self.matcher(input_dict)

        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()

        return mkpts0, mkpts1
        # Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
        # inliers = inliers > 0

        # draw_LAF_matches(
        #     KF.laf_from_center_scale_ori(
        #         torch.from_numpy(mkpts0).view(1, -1, 2),
        #         torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
        #         torch.ones(mkpts0.shape[0]).view(1, -1, 1),
        #     ),
        #     KF.laf_from_center_scale_ori(
        #         torch.from_numpy(mkpts1).view(1, -1, 2),
        #         torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
        #         torch.ones(mkpts1.shape[0]).view(1, -1, 1),
        #     ),
        #     torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
        #     K.tensor_to_image(img1),
        #     K.tensor_to_image(img2),
        #     inliers,
        #     draw_dict={"inlier_color": (0.2, 1, 0.2), "tentative_color": None, "feature_color": (0.2, 0.5, 1), "vertical": False},
        # )

        # plt.show()
        