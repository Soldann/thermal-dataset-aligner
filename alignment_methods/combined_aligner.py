from .dataset_aligner_base import DatasetAligner
from .contour_aligner import ContourAligner
from .feature_aligner import FeatureAligner
import numpy as np

class CombinedAligner(DatasetAligner):
    """Combines FeatureAligner and ContourAligner for robust alignment. This uses FeatureAligner unless the proposed translation exceeds the given threshold, in which case it falls back to ContourAligner."""
    def __init__(self, threshold=15.0):
        super().__init__()
        self.threshold = threshold
        self.feature_aligner = FeatureAligner()
        self.contour_aligner = ContourAligner()

    def align_images(self, rgb_image, thermal_image):
        # Use feature aligner first
        initial_translation = self.feature_aligner.align_images(rgb_image, thermal_image)

        if np.linalg.norm(initial_translation, ord=np.inf) > self.threshold:
            # If the initial translation is too large, use contour aligner
            refined_translation = self.contour_aligner.align_images(rgb_image, thermal_image)
            return refined_translation
        else:
            return initial_translation
