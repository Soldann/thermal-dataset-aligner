



from enum import Enum
from typing import List
from typing_extensions import Annotated
from pathlib import Path
import tyro
from dataclasses import dataclass
from alignment_methods.contour_aligner import ContourAligner
from alignment_methods.feature_aligner import FeatureAligner
from alignment_methods.dataset_aligner_base import DatasetAligner
from feature_matchers.base_feature_matcher import FeatureMatcher
from feature_matchers.vggt_feature_matcher import VGGTFeatureMatcher
import cv2
from vggt.utils.load_fn import load_and_preprocess_images
from dataset_aligner import AlignmentMethod, DatasetFormat
from plotting import make_matching_figure, create_interactive_correspondence_plot_from_kpts, plot_correspondences
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt

# Set matplotlib backend for GUI environments
plt.switch_backend('tkagg')

class ImageModality(Enum):
    thermal = "thermal"
    rgb = "rgb"

class DatasetCorrespondencer:
    def __init__(self, feature_matcher, dataset_aligner):
        self.feature_matcher: FeatureMatcher = feature_matcher
        self.dataset_aligner: DatasetAligner = dataset_aligner

    def compute_correspondences(self, rgb_images: List[Path], thermal_images: List[Path], original_image_modalities: List[ImageModality]):
        """"
        Compute correspondences between two images from the given paths. Currently only supports passing in two images at a time.
        :param rgb_images: List of paths to the two RGB images
        :param thermal_images: List of paths to the two thermal images
        :param original_image_modalities: List indicating which modality each image originally is
        :return: keypoints in image 1, keypoints in image 2, confidence scores
        """
        processed_rgb_images = load_and_preprocess_images(rgb_images)[:2]  # only keep 2 images
        kpts1, kpts2, conf = self.feature_matcher.compute_correspondences_from_tensor(processed_rgb_images)

        processed_thermal_images = load_and_preprocess_images(thermal_images)[:2]  # only keep 2 images
        print(original_image_modalities)
        if original_image_modalities[0] == ImageModality.thermal:
            og_image1 = processed_thermal_images[0].permute(1, 2, 0).cpu().numpy()

            tx, ty = self.dataset_aligner.align_images(processed_rgb_images[0].permute(1, 2, 0).cpu().numpy(), og_image1)
            kpts2 = kpts2 + np.round(np.array([tx, ty])).astype(np.int32)
        else:
            og_image1 = processed_rgb_images[0].permute(1, 2, 0).cpu().numpy()

        if original_image_modalities[1] == ImageModality.thermal:
            og_image2 = processed_thermal_images[1].permute(1, 2, 0).cpu().numpy()

            tx, ty = self.dataset_aligner.align_images(processed_rgb_images[1].permute(1, 2, 0).cpu().numpy(), og_image2)
            kpts1 = kpts1 + np.round(np.array([tx, ty])).astype(np.int32)
        else:
            og_image2 = processed_rgb_images[1].permute(1, 2, 0).cpu().numpy()

        plot_correspondences(
            og_image1,
            og_image2,
            kpts1,
            kpts2,
            draw_lines=False,
        )
        create_interactive_correspondence_plot_from_kpts(og_image1, og_image2, kpts1, kpts2, conf)
        color = cm.jet(conf[0, 0][kpts1[:,1], kpts2[:,0]] / conf.max())
        make_matching_figure(
            og_image1,
            og_image2,
            kpts1,
            kpts2,
            color=color,
        )
        plt.show()
        return kpts1, kpts2, conf

    def compute_correspondences_from_RGBT_Scenes(self, image_paths: List[Path]):
        rgb_images = []
        thermal_images = []
        original_image_modalities = []
        for path in image_paths:
            if "rgb" in str(path):
                rgb_images.append(path)
                thermal_images.append(path.parent.parent.parent / "thermal" / path.parent.name / path.name)
                original_image_modalities.append(ImageModality.rgb)
            else:
                rgb_images.append(path.parent.parent.parent / "rgb" / path.parent.name / path.name)
                thermal_images.append(path)
                original_image_modalities.append(ImageModality.thermal)
        return self.compute_correspondences(rgb_images, thermal_images, original_image_modalities)

@dataclass
class DatasetCorrespondencerConfigurator:
    image1_path: Annotated[Path, tyro.conf.Positional]
    image2_path: Annotated[Path, tyro.conf.Positional]
    dataset_format: DatasetFormat = DatasetFormat.RGBT_Scenes
    alignment_method: AlignmentMethod = AlignmentMethod.feature
    debug_mode: bool = False

    def main(self):
        correspondencer = DatasetCorrespondencer(VGGTFeatureMatcher(), self.alignment_method.value())
        # correspondencer.feature_matcher.debug_mode = self.debug_mode
        if self.dataset_format == DatasetFormat.RGBT_Scenes:
            kpts1, kpts2, conf = correspondencer.compute_correspondences_from_RGBT_Scenes([self.image1_path, self.image2_path])


if __name__ == "__main__":
    converter: DatasetCorrespondencerConfigurator = tyro.cli(DatasetCorrespondencerConfigurator)
    converter.main()
