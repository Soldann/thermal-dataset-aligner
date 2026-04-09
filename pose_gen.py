import cv2
import pycolmap
import numpy as np

from typing import List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
from dataset_correspondencer import DatasetCorrespondencer, ImageModality
from vggt.utils.load_fn import load_and_preprocess_images
import torch
from pathlib import Path
import kornia as K
import kornia.feature as KF
from alignment_methods.combined_aligner import CombinedAligner
from alignment_methods.dataset_aligner_base import DatasetAligner
from feature_matchers.base_feature_matcher import FeatureMatcher
from feature_matchers.vggt_feature_matcher import VGGTFeatureMatcher
import ahrs
import glob
import tyro
from tyro.conf import Positional
from dataclasses import dataclass, field
from typing import Literal, Optional
from typing_extensions import Annotated
import json

def visualize_points_3d(points_xyz, figsize=(8, 8), elev=20, azim=-60):
    """
    points_xyz: (N, 3) array-like of [x, y, z] points in world coordinates.
    """
    points_xyz = np.asarray(points_xyz)
    assert points_xyz.shape[1] == 3, "points_xyz must be of shape (N, 3)"

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    ax.scatter(x, y, z, s=1, c='k', alpha=0.7)

    # Equal aspect ratio
    max_range = (points_xyz.max(axis=0) - points_xyz.min(axis=0)).max()
    mid = points_xyz.mean(axis=0)
    for axis, m in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid):
        axis(m - max_range / 2, m + max_range / 2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.show()

import numpy as np
from scipy.spatial import cKDTree

def build_kdtree(image):
    coords = np.array([kp.xy for kp in image.points2D])
    return cKDTree(coords), coords


def find_keypoints_vectorized(kdtree, coords, uv_array, max_dist=3.0):
    """
    uv_array: shape (N, 2) array of (u, v) coordinates
    Returns:
        idxs: keypoint indices (N,)
        mask: boolean mask where True = valid match within max_dist
    """
    dists, idxs = kdtree.query(uv_array, k=1)
    mask = dists <= max_dist
    return idxs, mask

@dataclass
class NerfstudioTransformWriter:
    """Converter for TartanAir/Ground dataset to NerfStudio format."""
    
    base_path: Positional[Path]
    """Path to the base folder of the dataset."""

    pose_limit: Annotated[Optional[int], tyro.conf.arg(aliases=["-p"])] = None
    """Limit the number of poses to convert."""
    uniform: Annotated[bool, tyro.conf.arg(aliases=["-u"])] = False
    """If True, distribute the poses uniformly."""
    output_path: Annotated[Optional[Path], tyro.conf.arg(aliases=["-o"])] = None
    """Path to the output transforms.json file. If None, it will be saved in the base path."""

    save_ground_truth_poses: Annotated[bool, tyro.conf.arg(aliases=["-g"])] = False
    """If True, save the ground truth poses from COLMAP into the transforms.json file instead of the estimated ones."""

    print_estimation_errors: Annotated[bool, tyro.conf.arg(aliases=["-e"])] = False
    """If True, print the translation and rotation estimation errors compared to COLMAP poses."""

    _current_dir: Path = field(default_factory=lambda: Path().absolute(), init=False)
    def write_transforms(self, intrinsics, poses):
        """
        Write the transforms.json file.
        :param intrinsics: Dictionary of camera intrinsics. Should include keys "camera_model", "fl_x", "fl_y", "cx", "cy", "w", "h".
        :param poses: List of poses to write.
        :return:
        """
        global available_intrinsics

        transforms = intrinsics.copy()
        transforms["frames"] = []

        for frame in poses:
            transforms["frames"].append(frame)

        with open(self.output_path, "w") as f:
            json.dump(transforms, f, indent=4)
            print(f'Transforms file written to {self.output_path}')

    def main(self):
        if self.output_path is None:
            self.output_path = self.base_path / "transforms.json"

        # Load the COLMAP model
        model_path = self.base_path / "colmap" / "sparse" / "0"
        model = pycolmap.Reconstruction(model_path)

        points = np.zeros((len(model.points3D), 3))
        # Extract camera poses
        for i, (point3D_id, point3D) in enumerate(model.points3D.items()):
            points[i] = point3D.xyz

        # print("3D Points from COLMAP:")
        # visualize_points_3d(points)

        data_correspondencer = None
        thermal_images: List[Path] = [Path(p) for p in glob.glob(str(self.base_path / "thermal" / "**" / "*.jpg"))]
        thermal_images.sort()  # Ensure consistent order

        keypoint_cache_path: Path = self.base_path / 'keypoint_cache' / "correspondences_rgb_to_thermal"
        (keypoint_cache_path).mkdir(parents=True, exist_ok=True)

        camera_model = model.cameras[model.find_image_with_name("img_249.jpg").camera_id]
        K = camera_model.calibration_matrix()

        intrinsics = {
            "camera_model": camera_model.model.name,
            "fl_x": camera_model.params[0],
            "fl_y": camera_model.params[1],
            "cx": camera_model.params[2],
            "cy": camera_model.params[3],
            "w": camera_model.width,
            "h": camera_model.height,
        }

        poses = []
        translation_errors = []
        rotation_errors = []
        for img_path in thermal_images:
            rgb_img = Path(str(img_path).replace("thermal", "rgb"))
            print(f"Processing image: {img_path} and {rgb_img}")
            if not (keypoint_cache_path / f"{img_path.stem}_correspondences.pt").exists():
                if data_correspondencer is None:
                    data_correspondencer = DatasetCorrespondencer(VGGTFeatureMatcher(), CombinedAligner())
                    print("Initialized VGGTFeatureMatcher")
                images = load_and_preprocess_images([img_path, rgb_img])
                _, _, kpts1, kpts2, conf = data_correspondencer.compute_correspondences([rgb_img]*2, [img_path]*2, [ImageModality.rgb, ImageModality.thermal])

                torch.save((kpts1, kpts2, conf), keypoint_cache_path / f"{img_path.stem}_correspondences.pt")
            else:
                print("loading cached correspondences for image:", img_path)
                kpts1, kpts2, conf = torch.load(keypoint_cache_path / f"{img_path.stem}_correspondences.pt", map_location='cpu')

            colmap_img = model.find_image_with_name(rgb_img.stem + rgb_img.suffix)
            kdtree, coords = build_kdtree(colmap_img)

            idxs, mask = find_keypoints_vectorized(kdtree, coords, kpts1)
            idx_3d = np.array([
                p.point3D_id
                for p in (colmap_img.points2D[i] for i in idxs[mask])
                if p.has_point3D()
            ])

            kpts_thermal = np.array([
                p.xy
                for p in (colmap_img.points2D[i] for i in idxs[mask])
                if p.has_point3D()
            ])

            print(idxs.shape, mask.shape)

            # print(np.array(colmap_img.points2D[idxs][mask].point3D_id))
            matched_points3D = np.array([model.points3D[pt3D_id].xyz for pt3D_id in idx_3d])
        
            # print(f"Number of matched 3D points for {img_path.stem}: {len(matched_points3D)}")
            # visualize_points_3d(matched_points3D)

            _, rvec, tvec, inliers = cv2.solvePnPRansac(matched_points3D, kpts_thermal , K, distCoeffs=None)
            # print(f"Estimated pose for {img_path.stem}:")
            # print("Rotation vector (rvec):", rvec.flatten())
            # print("Translation vector (tvec):", tvec.flatten())

            R_est, _ = cv2.Rodrigues(rvec)

            # Get ground truth poses
            gt_pose = colmap_img.cam_from_world().matrix()
            R_gt = gt_pose[:3, :3]
            t_gt = gt_pose[:3, 3]


            # Convert to homogeneous transformation matrix for NerfStudio
            nerf_pose = np.eye(4)
            if not self.save_ground_truth_poses:
                nerf_pose[:3, :3] = R_est
                nerf_pose[:3, 3] = tvec.flatten()
            else:
                nerf_pose[:3, :3] = R_gt
                nerf_pose[:3, 3] = t_gt.flatten()
            frame = {
                "file_path": f"{img_path.relative_to(self.base_path)}",
                "transform_matrix": nerf_pose.tolist()
            }
            poses.append(frame)
            
            if self.print_estimation_errors:
                translation_error = np.linalg.norm(t_gt - tvec.flatten())
                print("Translation error (L2 norm):", translation_error)
                translation_errors.append(translation_error)
                # Compute yaw error
                quaternion_angle_diff = ahrs.utils.metrics.qad(ahrs.Quaternion(dcm=R_est), ahrs.Quaternion(dcm=R_gt))
                print("Rotation error (quaternion angle difference in degrees):", np.degrees(quaternion_angle_diff))
                rotation_errors.append(quaternion_angle_diff)   

        if self.print_estimation_errors:
            print(f"Average translation error across all images: {np.mean(translation_errors)}")
            print(f"Average rotation error across all images (degrees): {np.mean(rotation_errors)}")

        # Write the transforms.json file
        self.write_transforms(intrinsics, poses)
            
        

if __name__ == "__main__":
    tyro.extras.set_accent_color('bright_yellow')
    generator: NerfstudioTransformWriter = tyro.cli(NerfstudioTransformWriter)
    generator.main()
