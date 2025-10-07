from .dataset_aligner_base import DatasetAligner
import numpy as np
import matplotlib.pyplot as plt 
import cv2

def chamfer_distance(edges1, edges2):
    edges1_bin = (edges1 > 0).astype(np.uint8) # Ensure binary masks
    edges2_bin = (edges2 > 0).astype(np.uint8) # Ensure binary masks
    dt1 = cv2.distanceTransform(1 - edges1_bin, cv2.DIST_L2, 3)
    dt2 = cv2.distanceTransform(1 - edges2_bin, cv2.DIST_L2, 3)

    # Average distance from edges1 to edges2 and vice versa
    dist1 = np.mean(dt2[edges1_bin > 0])
    dist2 = np.mean(dt1[edges2_bin > 0])
    return (dist1 + dist2) / 2

class ContourAligner(DatasetAligner):
    def __init__(self):
        self.debug_mode = False

    def optimize_translation(self, image_1, image_2, lr=1.0, iters=100, loss_fn=chamfer_distance):
        tx, ty = 0.0, 0.0
        eps = 1
        for _ in range(iters):
            current_distance = loss_fn(cv2.warpAffine(image_1, np.float32([[1, 0, tx], [0, 1, ty]]), (image_2.shape[1], image_2.shape[0]), flags=cv2.INTER_LINEAR), image_2)
            # Numerical gradient approximation
            d_tx = (loss_fn(cv2.warpAffine(image_1, np.float32([[1, 0, tx+eps], [0, 1, ty]]), (image_2.shape[1], image_2.shape[0]), flags=cv2.INTER_LINEAR), image_2) - current_distance) / eps
            d_ty = (loss_fn(cv2.warpAffine(image_1, np.float32([[1, 0, tx], [0, 1, ty+eps]]), (image_2.shape[1], image_2.shape[0]), flags=cv2.INTER_LINEAR), image_2) - current_distance) / eps
            # Update translation
            tx -= lr * d_tx
            ty -= lr * d_ty

            #visualize the translated photos
            if _ % 10 == 0 and self.debug_mode:
                # Visualize translated c1 and c2 in different colours
                h, w = image_2.shape[:2]
                translated_c1 = cv2.warpAffine(image_1, np.float32([[1, 0, tx+eps], [0, 1, ty]]), (w, h))

                # Ensure binary masks
                mask1 = (translated_c1 > 0)
                mask2 = (image_2 > 0)

                # Create draw the masks on a canvas (RGB)
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                canvas[mask1] = (0, 0, 255) # Blue for image 1
                canvas[mask2] = (0, 255, 0) # Green for image 2
                overlap = mask1 & mask2
                canvas[overlap] = (255, 255, 0) # Yellow for the overlap

                plt.figure()
                plt.imshow(canvas)
                plt.title(f'Iteration {_+1}')
                print(d_tx, d_ty, current_distance)
                plt.show()
        return tx, ty

    def align_images(self, rgb_image, thermal_image):
        # Implement contour-based alignment logic here

        rgb_edges = cv2.Canny(rgb_image, 1, 200)
        thermal_edges = cv2.Canny(thermal_image, 30, 200)
        return self.optimize_translation(rgb_edges, thermal_edges)
