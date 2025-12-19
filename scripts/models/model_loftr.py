from pathlib import Path
import kornia as K
import kornia.feature as KF
# from kornia_moons.viz import draw_LAF_matches
# import matplotlib.pyplot as plt
import torch

class ModelLoFTR:
    def __init__(self, device):
        super().__init__()
        # Get default configurations
        self.matcher = KF.LoFTR(pretrained="outdoor")

    def __call__(self, img1, img2):
        input_dict = {
            "image0": K.color.rgb_to_grayscale(img1),  # LofTR works on grayscale images only
            "image1": K.color.rgb_to_grayscale(img2),
        }

        with torch.inference_mode():
            correspondences = self.matcher(input_dict)

        mkpts0 = correspondences["keypoints0"].cpu().numpy()
        mkpts1 = correspondences["keypoints1"].cpu().numpy()

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
        