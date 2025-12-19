from .dataset_aligner_base import DatasetAligner
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from xoftr.utils.plotting import make_matching_figure
from xoftr.xoftr import XoFTR
from xoftr.config.default import get_cfg_defaults
from xoftr.utils.data_io import DataIOWrapper, lower_config

def compute_translation_with_ransac(points1, points2, threshold = 2.0, max_iter=1000):
    best_inlier_mask = []
    best_t = None

    for _ in range(max_iter):
        i = np.random.randint(len(points1))
        t_candidate = points2[i] - points1[i]
        transformed = points1 + t_candidate
        errors = np.linalg.norm(transformed - points2, axis=1)
        inlier_mask = (errors < threshold)

        if np.sum(inlier_mask) > np.sum(best_inlier_mask):
            best_inlier_mask = inlier_mask
            best_t = t_candidate

    return best_t, best_inlier_mask

class FeatureAligner(DatasetAligner):
    def __init__(self):
        super().__init__()
        # Get default configurations
        config = get_cfg_defaults(inference=True)
        config = lower_config(config)

        # Coarse level threshold
        config['xoftr']['match_coarse']['thr'] = 0.3 # Default 0.3

        # Fine level threshold
        config['xoftr']['fine']['thr'] = 0.1 # Default 0.1

        # It is posseble to get denser matches
        # If True, xoftr returns all fine-level matches for each fine-level window (at 1/2 resolution)
        config['xoftr']['fine']['denser'] = False # Default False

        # XoFTR model
        matcher = XoFTR(config=config["xoftr"])

        # The input image sizes for xoftr
        # Note: The output matches and output images are in original image size
        config['test']['img0_resize'] = 640 # resize the longer side, None for no resize
        config['test']['img1_resize'] = 640 # resize the longer side, None for no resize

        # The path for weights
        ckpt = Path(__file__).parent.parent / "XoFTR/notebooks/weights/weights_xoftr_640.ckpt"

        # Data I/O wrapper
        self.matcher = DataIOWrapper(matcher, config=config["test"], ckpt=ckpt)

    def align_images(self, rgb_image, thermal_image, inlier_method='T'):
        # Implement feature-based alignment logic here
        output_data = self.matcher.from_cv_imgs(thermal_image, rgb_image)

        # Matched keypoints
        mkpts0 = output_data['mkpts0']
        mkpts1 = output_data['mkpts1']

        # Confidence values for fine-level matching
        mconf = output_data['mconf']

        # Original images BGR or GRAY
        img0 = output_data['img0']
        img1 = output_data['img1']

        # Mask outliers using RANSAC (Homography or Fundamental Matrix)
        # RANSAC types: https://opencv.org/blog/evaluating-opencvs-new-ransacs/

        # if inlier_method == 'F':
        #     F, inlier_mask = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, ransacReprojThreshold=1, maxIters=10000, confidence=0.9999)
        # elif inlier_method == 'H':
        #     H_pred, inlier_mask = cv2.findHomography(mkpts0, mkpts1, cv2.USAC_MAGSAC, ransacReprojThreshold=1, maxIters=10000, confidence=0.9999)
        if inlier_method == 'A':
            a, inlier_mask = cv2.estimateAffinePartial2D(mkpts0, mkpts1, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=2000, confidence=0.99)
            print(a)
            t = a[:,2]
        elif inlier_method == 'T':
            t, inlier_mask = compute_translation_with_ransac(mkpts0, mkpts1, threshold=2.0, max_iter=2000)

        inlier_mask = inlier_mask.ravel() > 0
        mkpts0 = mkpts0[inlier_mask]
        mkpts1 = mkpts1[inlier_mask]
        mconf = mconf[inlier_mask]

        if self.debug_mode:
            # Draw
            color = cm.jet(mconf)
            text = [
                'XoFTR',
                'Matches: {}'.format(len(mconf)),
            ]
            if len(img0.shape) == 3:
                _img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
            else:
                _img0 = img0
            if len(img1.shape) == 3:
                _img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            else:
                _img1 = img1
            fig_org = make_matching_figure(_img0, _img1, np.zeros(0),  np.zeros(0),  np.zeros(0), text=["Original"], dpi=125)
            fig_match = make_matching_figure(_img0, _img1, mkpts0, mkpts1, color, text=text, dpi=125)

        return t
