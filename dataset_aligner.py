from enum import Enum
from typing import List
from typing_extensions import Annotated
from pathlib import Path
import tyro
from dataclasses import dataclass
from alignment_methods.contour_aligner import ContourAligner
from alignment_methods.feature_aligner import FeatureAligner

class DatasetFormat(Enum):
    RGBT_Scenes = "RGBT-Scenes"

class ImageCategory(Enum):
    train = "train"
    test = "test"
    all = "all"

class AlignmentMethod(Enum):
    contour = ContourAligner
    feature = FeatureAligner

@dataclass
class DatasetAlignerConfigurator:
    dataset_path: Annotated[Path, tyro.conf.Positional]
    dataset_format: DatasetFormat
    alignment_method: AlignmentMethod
    image_type: ImageCategory = ImageCategory.train

    def align_datasets(self, rgb_images: List[Path], thermal_images: List[Path]):
        # Implement alignment logic here
        print(f"Aligning {len(rgb_images)} RGB images with {len(thermal_images)} thermal images.")


    def main(self):
        self.dataset_aligner = self.alignment_method.value()

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

        self.align_datasets(rgb_images, thermal_images)
        self.dataset_aligner.align_images(rgb_images[0], thermal_images[0])

if __name__ == "__main__":
    converter: DatasetAlignerConfigurator = tyro.cli(DatasetAlignerConfigurator)
    converter.main()