from enum import Enum
from typing import List
from typing_extensions import Annotated
from pathlib import Path
import tyro
from dataclasses import dataclass
from alignment_methods.contour_aligner import ContourAligner
from alignment_methods.feature_aligner import FeatureAligner
from alignment_methods.combined_aligner import CombinedAligner
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Set matplotlib backend for GUI environments
plt.switch_backend('tkagg')

class DatasetFormat(Enum):
    RGBT_Scenes = "RGBT-Scenes"

class ImageCategory(Enum):
    train = "train"
    test = "test"
    all = "all"

class AlignmentMethod(Enum):
    contour = ContourAligner
    feature = FeatureAligner
    combined = CombinedAligner

@dataclass
class DatasetAlignerConfigurator:
    dataset_path: Annotated[Path, tyro.conf.Positional]
    dataset_format: DatasetFormat
    alignment_method: AlignmentMethod
    output_path: Path = Path("./aligned_output")
    image_type: ImageCategory = ImageCategory.train
    debug_mode: bool = False

    def align_datasets(self, rgb_images: List[Path], thermal_images: List[Path]):
        # Implement alignment logic here
        print(f"Aligning {len(rgb_images)} RGB images with {len(thermal_images)} thermal images.")
        for i in range(len(rgb_images)):
            print(f"Aligning pair {i+1}: {rgb_images[i]} and {thermal_images[i]}")
            rgb_image = cv2.imread(str(rgb_images[i]), cv2.IMREAD_COLOR)
            thermal_image = cv2.imread(str(thermal_images[i]), cv2.IMREAD_COLOR)
            tx, ty = self.dataset_aligner.align_images(rgb_image, thermal_image)
            print(f"Computed translation: {tx}, {ty}")

            M = np.float32([[1, 0, tx], [0, 1, ty]])
            translated_therm = cv2.warpAffine(thermal_image, M, (rgb_image.shape[1], rgb_image.shape[0]))
            blended = cv2.addWeighted(rgb_image, 0.5, translated_therm, 0.5, 0)  # Blend the two images
            plt.figure()
            plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
            if self.debug_mode:
                plt.title(f'Blended Image after Translation')
                plt.show()
            self.output_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(self.output_path /f'aligned_{i+1}.png')


    def main(self):
        self.dataset_aligner = self.alignment_method.value()
        self.dataset_aligner.debug_mode = self.debug_mode

        if self.dataset_format == DatasetFormat.RGBT_Scenes:
            rgb_dir = Path(self.dataset_path) / 'rgb'
            thermal_dir = Path(self.dataset_path) / 'thermal'
            rgb_images = []
            thermal_images = []
            if self.image_type == ImageCategory.train or self.image_type == ImageCategory.all:
                rgb_images.extend((rgb_dir / 'train').glob('*.jpg'))
                thermal_images.extend((thermal_dir / 'train').glob('*.jpg'))
            if self.image_type == ImageCategory.test or self.image_type == ImageCategory.all:
                rgb_images.extend((rgb_dir / 'test').glob('*.jpg'))
                thermal_images.extend((thermal_dir / 'test').glob('*.jpg'))
            assert len(rgb_images) == len(thermal_images), "Mismatch in number of RGB and thermal images"
            rgb_images = sorted(rgb_images)
            thermal_images = sorted(thermal_images)
        self.align_datasets(rgb_images, thermal_images)

if __name__ == "__main__":
    converter: DatasetAlignerConfigurator = tyro.cli(DatasetAlignerConfigurator)
    converter.main()