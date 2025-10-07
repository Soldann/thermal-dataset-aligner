import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_bgr(img, title=None, cmap=None):
    if img is None:
        return
    # If single channel, display with gray colormap
    if len(img.shape) == 2 or img.shape[2] == 1:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if title:
        plt.title(title)
    plt.axis('off')

image = cv2.imread('/home/landson/RGBT-Scenes/Building/rgb/test/img_001.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 1, 200)
contours, hierarchy = cv2.findContours(edged,
                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
plt.figure()
show_bgr(edged, 'Canny Edges (RGB)')

image_therm = cv2.imread('/home/landson/RGBT-Scenes/Building/thermal/test/img_001.jpg')
gray = cv2.cvtColor(image_therm, cv2.COLOR_BGR2GRAY)
edged_therm = cv2.Canny(gray, 30, 200)
contours_therm, hierarchy = cv2.findContours(edged_therm,
                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
plt.figure()
show_bgr(edged_therm, 'Canny Edges (Thermal)')

# Translate the images so that the contours are aligned
def translate_contour(contour, tx, ty):
    translation_matrix = np.array([[1, 0, tx], [0, 1, ty]])
    ones = np.ones((contour.reshape(-1, 2).shape[0], 1))
    points_ones = np.hstack([contour.reshape(-1, 2), ones])
    translated_points = points_ones @ translation_matrix.T
    return translated_points[:, :2].reshape(contour.shape)


def chamfer_distance(edges1, edges2):
    edges1_bin = (edges1 > 0).astype(np.uint8) # Ensure binary masks
    edges2_bin = (edges2 > 0).astype(np.uint8) # Ensure binary masks
    dt1 = cv2.distanceTransform(1 - edges1_bin, cv2.DIST_L2, 3)
    dt2 = cv2.distanceTransform(1 - edges2_bin, cv2.DIST_L2, 3)

    # Average distance from edges1 to edges2 and vice versa
    dist1 = np.mean(dt2[edges1_bin > 0])
    dist2 = np.mean(dt1[edges2_bin > 0])
    return (dist1 + dist2) / 2

def optimize_translation(c1, c2, lr=1.0, iters=100, loss_fn=chamfer_distance):
    tx, ty = 0.0, 0.0
    eps = 1
    for _ in range(iters):
        current_distance = loss_fn(cv2.warpAffine(c1, np.float32([[1, 0, tx], [0, 1, ty]]), (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR), c2)
        # Numerical gradient approximation
        d_tx = (loss_fn(cv2.warpAffine(c1, np.float32([[1, 0, tx+eps], [0, 1, ty]]), (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR), c2) - current_distance) / eps
        d_ty = (loss_fn(cv2.warpAffine(c1, np.float32([[1, 0, tx], [0, 1, ty+eps]]), (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR), c2) - current_distance) / eps
        # Update translation
        tx -= lr * d_tx
        ty -= lr * d_ty

        #visualize the translated photos
        if _ % 10 == 0:
            # Visualize translated c1 and c2 in different colours
            h, w = image.shape[:2]
            translated_c1 = cv2.warpAffine(c1, np.float32([[1, 0, tx+eps], [0, 1, ty]]), (w, h))

            # Ensure binary masks
            mask1 = (translated_c1 > 0)
            mask2 = (c2 > 0)

            # Create a color canvas (BGR)
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            # translated_c1 -> green (0,255,0), c2 -> blue (255,0,0)
            canvas[mask1] = (0, 255, 0)
            canvas[mask2] = (255, 0, 0)
            # Overlap -> yellow (0,255,255) in BGR
            overlap = mask1 & mask2
            canvas[overlap] = (0, 255, 255)

            plt.figure()
            show_bgr(canvas, f'Iteration {_+1}')
            print(d_tx, d_ty, current_distance)
            plt.show()
    return tx, ty


def soft_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return cv2.normalize(magnitude, None, 0, 1.0, cv2.NORM_MINMAX)

def edge_l2_loss(e1, e2):
    return np.mean((e1 - e2) ** 2)


soft_edged_therm = soft_edges(image_therm)
soft_edged = soft_edges(image)

plt.figure()
show_bgr(soft_edged, 'Soft Edges (RGB)')
plt.figure()
show_bgr(soft_edged_therm, 'Soft Edges (Thermal)')
plt.show()

tx, ty = optimize_translation(edged_therm, edged, lr=1, iters=100, loss_fn=chamfer_distance)

# Select the largest contour from each image for alignment
# c1 = max(contours, key=cv2.contourArea)
# c2 = max(contours_therm, key=cv2.contourArea)
# Run ICP to find the best translation
# tx, ty = optimize_translation(edged, edged_therm, lr=0.1, iters=10)
print(f"Optimal translation: tx={tx}, ty={ty}")
# Translate the thermal contour
# aligned_c1 = translate_contour(c1, tx, ty)

# Draw contours for visualization
# canvas = np.zeros_like(image)
# cv2.drawContours(canvas, [c1], -1, (0, 255, 0), 1)  # Aligned contour in green
# cv2.drawContours(canvas, [c2], -1, (255, 0, 0), 1)  # Thermal contour in blue
# plt.figure()
# show_bgr(canvas, 'Original Contours')
# print(f"Number of contours in RGB image: {len(contours)}")
# print(f"Number of contours in Thermal image: {len(contours_therm)}")


# canvas = np.zeros_like(image)
# cv2.drawContours(canvas, [aligned_c1.astype(np.int32)], -1, (0, 255, 0), 1)  # Aligned contour in green
# cv2.drawContours(canvas, [c2], -1, (255, 0, 0), 1)  # Thermal contour in blue
# plt.figure()
# show_bgr(canvas, 'Aligned Contours')

# Draw the translated thermal image on top of the RGB image
M = np.float32([[1, 0, tx], [0, 1, ty]])
translated_therm = cv2.warpAffine(image_therm, M, (image.shape[1], image.shape[0]))
blended = cv2.addWeighted(image, 0.5, translated_therm, 0.5, 0)  # Blend the two images
plt.figure()
show_bgr(blended, 'Blended Image')

# Show the original images for comparison
plt.figure()
show_bgr(image, 'Original RGB Image')
plt.figure()
blended_therm_original = cv2.addWeighted(image, 0.5, image_therm, 0.5, 0)
show_bgr(blended_therm_original, 'Original Blended Image')

plt.show()
