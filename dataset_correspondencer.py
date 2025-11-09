



from enum import Enum
from typing import List
import torch
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
    def __init__(self, feature_matcher: FeatureMatcher, dataset_aligner: DatasetAligner, debug_mode: bool = False):
        self.feature_matcher: FeatureMatcher = feature_matcher
        self.dataset_aligner: DatasetAligner = dataset_aligner
        self.debug_mode = debug_mode

    def compute_correspondences(self, rgb_images: List[Path], thermal_images: List[Path], original_image_modalities: List[ImageModality], correspondence_pairs: List[tuple] = [(0,1)]):
        """
        Compute correspondences between a set of images that have both RGB and thermal modalities.
        :param rgb_images: List of paths to the RGB images
        :param thermal_images: List of paths to the thermal images
        :param original_image_modalities: List indicating the target modality of each image we want to compute correspondences for
        :param correspondence_pairs: List of tuples indicating which image pairs to compute correspondences for
        :return: tensor of image1, tensor of image2, keypoints in image 1, keypoints in image 2, confidence scores
        """
        processed_rgb_images = load_and_preprocess_images(rgb_images)
        kpts1, kpts2, conf = self.feature_matcher.compute_correspondences_from_tensor(processed_rgb_images, correspondence_pairs)
        processed_thermal_images = load_and_preprocess_images(thermal_images)
        print(original_image_modalities)

        og_image1 = []
        og_image2 = []

        for i, (image1_idx, image2_idx) in enumerate(correspondence_pairs):
            if original_image_modalities[image1_idx] == ImageModality.thermal:
                og_image1.append(processed_thermal_images[image1_idx])

                tx, ty = self.dataset_aligner.align_images(processed_rgb_images[image1_idx].permute(1, 2, 0).cpu().numpy(), og_image1[-1].permute(1, 2, 0).cpu().numpy())
                kpts2[i] = kpts2[i] + torch.round(torch.tensor([tx, ty])).to(torch.int32)
            else:
                og_image1.append(processed_rgb_images[image1_idx])

            if original_image_modalities[image2_idx] == ImageModality.thermal:
                og_image2.append(processed_thermal_images[image2_idx])

                tx, ty = self.dataset_aligner.align_images(processed_rgb_images[image2_idx].permute(1, 2, 0).cpu().numpy(), og_image2[-1].permute(1, 2, 0).cpu().numpy())
                kpts1[i] = kpts1[i] + torch.round(torch.tensor([tx, ty])).to(torch.int32)
            else:
                og_image2.append(processed_rgb_images[image2_idx])

            if self.debug_mode:
                og_image1_np = og_image1[i].permute(1, 2, 0).cpu().numpy()
                og_image2_np = og_image2[i].permute(1, 2, 0).cpu().numpy()
                kpts1_np = kpts1[i].cpu().numpy()
                kpts2_np = kpts2[i].cpu().numpy()
                conf_np = conf[i].cpu().numpy()

                plot_correspondences(
                    og_image1_np,
                    og_image2_np,
                    kpts1_np,
                    kpts2_np,
                    draw_lines=False,
                )
                create_interactive_correspondence_plot_from_kpts(og_image1_np, og_image2_np, kpts1_np, kpts2_np, conf_np)
                color = cm.jet(conf_np[0, 0][kpts1_np[:,1], kpts2_np[:,0]] / conf_np.max())
                text = [
                    'Ours',
                    'Matches: {}'.format(len(kpts1_np)),
                ]
                make_matching_figure(
                    og_image1_np,
                    og_image2_np,
                    kpts1_np,
                    kpts2_np,
                    color=color,
                    text=text,
                )
                plt.show()
        return og_image1, og_image2, kpts1, kpts2, conf

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
        correspondencer = DatasetCorrespondencer(VGGTFeatureMatcher(), self.alignment_method.value(), self.debug_mode)
        # correspondencer.feature_matcher.debug_mode = self.debug_mode
        if self.dataset_format == DatasetFormat.RGBT_Scenes:
            og_image1, og_image2, kpts1, kpts2, conf = correspondencer.compute_correspondences_from_RGBT_Scenes([self.image1_path, self.image2_path])


if __name__ == "__main__":
    converter: DatasetCorrespondencerConfigurator = tyro.cli(DatasetCorrespondencerConfigurator)
    converter.main()
