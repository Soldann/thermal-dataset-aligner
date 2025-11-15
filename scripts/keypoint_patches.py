import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import torch

def visualize_patch_matches(img1, img2, matches, patch_size=14, color=(0, 255, 0)):
    # Resize images to same height
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    if h1 != h2:
        scale = h1 / h2
        img2 = cv2.resize(img2, (int(w2 * scale), h1))

    # Convert images to uint8 if they are not
    if img1.dtype != np.uint8:
        img1 = (img1 * 255).astype(np.uint8)
    if img2.dtype != np.uint8:
        img2 = (img2 * 255).astype(np.uint8)

    num_cols = w1 // patch_size
    # Concatenate images side by side
    vis_img = np.concatenate([img1, img2], axis=1)
    offset = img1.shape[1]

    print("img1 shape:", img1.shape, img1.dtype)
    print("img2 shape:", img2.shape, img2.dtype)
    print("vis_img dtype:", vis_img.dtype)
    print("vis_img shape:", vis_img.shape)
    vis_img = np.ascontiguousarray(vis_img)

    for (patch_1, patch_2) in matches[:10]:
        patch_1_row = patch_1 // num_cols
        patch_1_col = patch_1 % num_cols
        patch_2_row = patch_2 // num_cols
        patch_2_col = patch_2 % num_cols
        print("Visualizing match:", patch_1, patch_2)
        pt1 = tuple((patch_1_col * patch_size + patch_size // 2, patch_1_row * patch_size + patch_size // 2)) # Center of patch 1
        pt2 = tuple((patch_2_col * patch_size + patch_size // 2, patch_2_row * patch_size + patch_size // 2)) # Center of patch 2
        pt2_shifted = (pt2[0] + offset, pt2[1]) # To concatenated image coordinates

        # Draw patch rectangles
        half = patch_size // 2
        print("Drawing rectangles at:", pt1, pt2, pt2_shifted)
        print("Rectangle corners 1:", (int(pt1[0] - half), int(pt1[1] - half)), (int(pt1[0] + half), int(pt1[1] + half)))
        print("Rectangle corners 2:", (int(pt2_shifted[0] - half), int(pt2_shifted[1] - half)), (int(pt2_shifted[0] + half), int(pt2_shifted[1] + half)))
        cv2.rectangle(vis_img, (int(pt1[0] - half), int(pt1[1] - half)), (int(pt1[0] + half), int(pt1[1] + half)), color, 1)
        cv2.rectangle(vis_img, (int(pt2_shifted[0] - half), int(pt2_shifted[1] - half)), (int(pt2_shifted[0] + half), int(pt2_shifted[1] + half)), color, 1)

        # Draw line connecting patches
        cv2.line(vis_img, pt1, pt2_shifted, color, 1)

    # Show image
    plt.figure(figsize=(12, 6))
    plt.imshow(vis_img)
    plt.axis('off')
    plt.title('Patch Matches')
    plt.show()

def compute_patch_matches(keypoints1, keypoints2, image_shape, patch_size=14):
    """
    Given a set of keypoint correspondences between two images, compute the patch-level matches.
    Each image is divided into patches of size patch_size x patch_size, and keypoints are
    assigned to patches based on their coordinates.
    :param keypoints1: Nx2 tensor of keypoints in image 1
    :param keypoints2: Nx2 tensor of keypoints in image 2
    :param image_shape: Tuple (H, W) of the image dimensions
    :param patch_size: Size of each patch (assumed square)
    :return: Two lists of patch indices corresponding to matches: patch1, patch2
    """
    H, W = image_shape
    num_cols = W // patch_size

    def to_patch_index(kp):
        x, y = kp
        row = int(y // patch_size)
        col = int(x // patch_size)
        return row * num_cols + col  # flatten to 1D index

    # Convert keypoints to patch indices
    patch_matches = []
    print(keypoints1.shape, keypoints2.shape)
    for kp1, kp2 in zip(keypoints1, keypoints2):
        p1 = to_patch_index(kp1)
        p2 = to_patch_index(kp2)
        if 0 <= p1 < (H // patch_size) * num_cols and 0 <= p2 < (H // patch_size) * num_cols:
            patch_matches.append((p1, p2))
        else:
            # print(f"Keypoint pair ({kp1}, {kp2}) is out of bounds.")
            continue

    # Count occurrences of each patch-pair
    match_counts = Counter(patch_matches)

    # Sort by frequency (descending)
    sorted_matches = sorted(match_counts.items(), key=lambda x: x[1], reverse=True)

    # Split into two lists
    patch1 = torch.tensor([p[0][0] for p in sorted_matches])
    patch2 = torch.tensor([p[0][1] for p in sorted_matches])

    return patch1, patch2  # List of (patch1, patch2)


def batched_compute_patch_matches(batch_kpts1, batch_kpts2, kpts1_length, kpts2_length, image_shape, patch_size=14):
    """
    Fully vectorized batched version using scatter_add.
    Keeps all nonzero matches per batch, pads to max length, returns mask.
    :param batch_kpts1: BxNx2 tensor of keypoints in image 1
    :param batch_kpts2: BxNx2 tensor of keypoints in image 2
    :param image_shape: Tuple (H, W) of the image dimensions
    :param patch_size: Size of each patch (assumed square)
    :return: patch1, patch2, mask
             patch1: [B, max_matches] tensor of patch indices in image 1
             patch2: [B, max_matches] tensor of patch indices in image 2
             mask:   [B, max_matches] boolean tensor
    """
    B, N, _ = batch_kpts1.shape
    H, W = image_shape
    num_cols = W // patch_size
    num_rows = H // patch_size
    max_index = num_rows * num_cols

    # Compute patch indices
    p1 = (batch_kpts1[:,:,1] // patch_size).long() * num_cols + (batch_kpts1[:,:,0] // patch_size).long()
    p2 = (batch_kpts2[:,:,1] // patch_size).long() * num_cols + (batch_kpts2[:,:,0] // patch_size).long()

    # Mask out-of-bounds
    mask_valid = (p1 >= 0) & (p1 < max_index) & (p2 >= 0) & (p2 < max_index) & (torch.arange(N).unsqueeze(0).to(kpts1_length.device) < kpts1_length.unsqueeze(1)) & (torch.arange(N).unsqueeze(0).to(kpts2_length.device) < kpts2_length.unsqueeze(1))

    # Encode pairs into single indices
    pair_index = torch.where(
        mask_valid,
        (p1 * max_index + p2).long(),   # valid pairs
        torch.full_like(p1, max_index*max_index)   # invalid pairs → sentinel
    )

    # Scatter counts into [B, max_index*max_index]
    counts = torch.zeros(B, max_index*max_index + 1, dtype=torch.long, device=pair_index.device)
    counts.scatter_add_(1, pair_index, torch.ones_like(pair_index, dtype=torch.long))
    counts = counts[:, :max_index*max_index]  # discard sentinel column

    # Sort by frequency per batch
    sorted_idx = torch.argsort(counts, dim=1, descending=True, stable=True)

    # Gather counts in sorted order
    sorted_counts = counts.gather(1, sorted_idx)

    # Mask: valid if count > 0
    mask = sorted_counts > 0

    # Decode back into (p1,p2)
    patch1 = sorted_idx // max_index
    patch2 = sorted_idx % max_index

    print("patch1 shape:", patch1.shape, patch1.dtype)

    # Pad dynamically: just keep full sorted list, mask tells you which are valid
    return patch1, patch2, mask