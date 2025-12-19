from pathlib import Path
from xoftr.utils.plotting import make_matching_figure
from xoftr.xoftr import XoFTR
from xoftr.config.default import get_cfg_defaults
from xoftr.utils.data_io import DataIOWrapper, lower_config

class ModelXoFTR:
    def __init__(self, device):
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
        ckpt = Path(__file__).parent.parent.parent / "XoFTR/notebooks/weights/weights_xoftr_640.ckpt"

        # Data I/O wrapper
        self.matcher = DataIOWrapper(matcher, config=config["test"], ckpt=ckpt)

    def __call__(self, img1, img2):
        # Implement feature-based alignment logic here
        output_data = self.matcher.from_cv_imgs(img1, img2)

        # Matched keypoints
        mkpts0 = output_data['mkpts0']
        mkpts1 = output_data['mkpts1']

        # Confidence values for fine-level matching
        mconf = output_data['mconf']

        # Original images BGR or GRAY
        img0 = output_data['img0']
        img1 = output_data['img1']

        return mkpts0, mkpts1
        # Mask outliers using RANSAC (Homography or Fundamental Matrix)
        # RANSAC types: https://opencv.org/blog/evaluating-opencvs-new-ransacs/

        # if inlier_method == 'F':
        #     F, inlier_mask = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, ransacReprojThreshold=1, maxIters=10000, confidence=0.9999)
        # elif inlier_method == 'H':
        #     H_pred, inlier_mask = cv2.findHomography(mkpts0, mkpts1, cv2.USAC_MAGSAC, ransacReprojThreshold=1, maxIters=10000, confidence=0.9999)
        