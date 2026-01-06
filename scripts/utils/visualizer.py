import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max([x_range, y_range, z_range])

    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)

    ax.set_xlim3d([mid_x - max_range/2, mid_x + max_range/2])
    ax.set_ylim3d([mid_y - max_range/2, mid_y + max_range/2])
    ax.set_zlim3d([mid_z - max_range/2, mid_z + max_range/2])

def visualize_pose_and_points(
    kpts1,
    kpts2,
    mask,
    K,
    R_gt,
    t_gt,
    R_est,
    t_est,
    title="Camera Pose and Triangulated 3D Points"
):
    """
    Visualize estimated vs ground-truth camera poses and triangulated 3D points.

    Parameters
    ----------
    kpts1, kpts2 : (N, 2) torch or numpy arrays
        Matched keypoints.
    mask : (N,) boolean mask
        Inlier mask from recoverPose.
    K : (3, 3) numpy array
        Camera intrinsic matrix.
    R_gt, R_est : (3, 3) numpy arrays
        Ground-truth and estimated rotation matrices. (from camera 2 to camera 1)
    t_gt, t_est : (3,) numpy arrays or torch tensors
        Ground-truth and estimated translations. (from camera 2 to camera 1)
    """

    # -----------------------------
    # Helper: draw a camera frame
    # -----------------------------
    def plot_camera(ax, R, t, color='blue', scale=0.1, label=None):
        C = t.reshape(3)

        x_axis = C + R[:, 0] * scale
        y_axis = C + R[:, 1] * scale
        z_axis = C + R[:, 2] * scale

        ax.plot([C[0], x_axis[0]], [C[1], x_axis[1]], [C[2], x_axis[2]], color='red')
        ax.plot([C[0], y_axis[0]], [C[1], y_axis[1]], [C[2], y_axis[2]], color='green')
        ax.plot([C[0], z_axis[0]], [C[1], z_axis[1]], [C[2], z_axis[2]], color='blue')

        ax.scatter(C[0], C[1], C[2], color=color, s=40)
        if label:
            ax.text(C[0], C[1], C[2], label, color=color)

    # -----------------------------
    # Helper: triangulation
    # -----------------------------
    def triangulate_points(k1, k2, K, R, t):
        proj1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        proj2 = K @ np.hstack([R, t.reshape(3, 1)])

        pts4d = cv2.triangulatePoints(proj1, proj2, k1.T, k2.T)
        pts3d = (pts4d[:3] / pts4d[3]).T
        return pts3d

    # -----------------------------
    # Convert inputs to numpy
    # -----------------------------
    if isinstance(kpts1, torch.Tensor):
        kpts1 = kpts1.cpu().numpy()
    if isinstance(kpts2, torch.Tensor):
        kpts2 = kpts2.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy().astype(bool)
    if isinstance(t_gt, torch.Tensor):
        t_gt = t_gt.cpu().numpy()
    if isinstance(t_est, torch.Tensor):
        t_est = t_est.cpu().numpy()

    # -----------------------------
    # Triangulate using inliers
    # -----------------------------
    pts3d = triangulate_points(
        kpts1[mask],
        kpts2[mask],
        K,
        R_est.T,  # Reverse the direction of the estimated pose to be consistent cv2 convention (transform cam1 to cam2)
        -R_est @ t_est.reshape(3,1)
    )

    # -----------------------------
    # Plot everything
    # -----------------------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Ground truth camera
    plot_camera(ax, R_gt, t_gt, color='green', label='GT Camera')

    # Estimated camera
    plot_camera(ax, R_est, t_est, color='red', label='Estimated Camera')

    # Origin camera
    plot_camera(ax, np.eye(3), np.zeros(3), color='blue', label='Origin')

    # Triangulated points
    ax.scatter(
        pts3d[:, 0], pts3d[:, 1], pts3d[:, 2],
        s=5, c='blue', alpha=0.5, label='Triangulated Points'
    )

    # Formatting
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    # Equal aspect ratio
    max_range = np.array([
        pts3d[:, 0].max() - pts3d[:, 0].min(),
        pts3d[:, 1].max() - pts3d[:, 1].min(),
        pts3d[:, 2].max() - pts3d[:, 2].min()
    ]).max()

    mid_x = (pts3d[:, 0].max() + pts3d[:, 0].min()) * 0.5
    mid_y = (pts3d[:, 1].max() + pts3d[:, 1].min()) * 0.5
    mid_z = (pts3d[:, 2].max() + pts3d[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

    # ax.autoscale(enable=False) # freeze limits
    # set_axes_equal(ax)
    plt.show()
