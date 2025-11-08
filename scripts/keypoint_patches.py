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

    # Concatenate images side by side
    vis_img = np.concatenate([img1, img2], axis=1)
    offset = img1.shape[1]

    for (pt1, pt2) in matches:
        pt1 = tuple(np.round(pt1).astype(int))
        pt2 = tuple(np.round(pt2).astype(int))
        pt2_shifted = (pt2[0] + offset, pt2[1])

        # Draw patch rectangles
        half = patch_size // 2
        cv2.rectangle(vis_img, (pt1[0] - half, pt1[1] - half), (pt1[0] + half, pt1[1] + half), color, 1)
        cv2.rectangle(vis_img, (pt2_shifted[0] - half, pt2_shifted[1] - half), (pt2_shifted[0] + half, pt2_shifted[1] + half), color, 1)

        # Draw line connecting patches
        cv2.line(vis_img, pt1, pt2_shifted, color, 1)

    # Show image
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Patch Matches')
    plt.show()

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
            print(f"Keypoint pair ({kp1}, {kp2}) is out of bounds.")

    # Count occurrences of each patch-pair
    match_counts = Counter(patch_matches)

    # Sort by frequency (descending)
    sorted_matches = sorted(match_counts.items(), key=lambda x: x[1], reverse=True)

    # Split into two lists
    patch1 = torch.tensor([p[0][0] for p in sorted_matches])
    patch2 = torch.tensor([p[0][1] for p in sorted_matches])

    return patch1, patch2  # List of (patch1, patch2)
