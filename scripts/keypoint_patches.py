import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import torch

def visualize_patch_matches(img1, img2, matches, patch_size=14, color=(255, 0, 0), patches_to_draw=10, headless=False):
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

    # print("img1 shape:", img1.shape, img1.dtype)
    # print("img2 shape:", img2.shape, img2.dtype)
    # print("vis_img dtype:", vis_img.dtype)
    # print("vis_img shape:", vis_img.shape)
    vis_img = np.ascontiguousarray(vis_img)

    for (patch_1, patch_2) in matches[:patches_to_draw]:
        patch_1_row = patch_1 // num_cols
        patch_1_col = patch_1 % num_cols
        patch_2_row = patch_2 // num_cols
        patch_2_col = patch_2 % num_cols
        # print("Visualizing match:", patch_1, patch_2)
        pt1 = tuple((patch_1_col * patch_size + patch_size // 2, patch_1_row * patch_size + patch_size // 2)) # Center of patch 1
        pt2 = tuple((patch_2_col * patch_size + patch_size // 2, patch_2_row * patch_size + patch_size // 2)) # Center of patch 2
        pt2_shifted = (pt2[0] + offset, pt2[1]) # To concatenated image coordinates

        # Draw patch rectangles
        half = patch_size // 2
        # print("Drawing rectangles at:", pt1, pt2, pt2_shifted)
        # print("Rectangle corners 1:", (int(pt1[0] - half), int(pt1[1] - half)), (int(pt1[0] + half), int(pt1[1] + half)))
        # print("Rectangle corners 2:", (int(pt2_shifted[0] - half), int(pt2_shifted[1] - half)), (int(pt2_shifted[0] + half), int(pt2_shifted[1] + half)))
        cv2.rectangle(vis_img, (int(pt1[0] - half), int(pt1[1] - half)), (int(pt1[0] + half), int(pt1[1] + half)), color, 1)
        cv2.rectangle(vis_img, (int(pt2_shifted[0] - half), int(pt2_shifted[1] - half)), (int(pt2_shifted[0] + half), int(pt2_shifted[1] + half)), color, 1)

        # Draw line connecting patches
        cv2.line(vis_img, pt1, pt2_shifted, color, 1)

    # Show image
    if not headless:
        plt.figure(figsize=(12, 6))
        plt.imshow(vis_img)
        plt.axis('off')
        plt.title('Patch Matches')
        plt.show()
    return vis_img

def convert_patches_to_keypoints(patch_indices, image_shape, patch_size=14):
    # Convert patch indices back to keypoint coordinates (center of patch)
    H, W = image_shape
    num_cols = W // patch_size
    keypoints = []
    for p in patch_indices:
        row = p // num_cols
        col = p % num_cols
        x = col * patch_size + patch_size // 2
        y = row * patch_size + patch_size // 2
        keypoints.append((x, y))
    return torch.tensor(keypoints)

def compute_patch_matches(keypoints1, keypoints2, image_shape, patch_size=14):
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
