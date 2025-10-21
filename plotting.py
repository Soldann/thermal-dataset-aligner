import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vggt.utils.load_fn import load_and_preprocess_images
import mplcursors
from matplotlib.patches import Circle
import matplotlib
from matplotlib.offsetbox import AnnotationBbox, TextArea

def create_interactive_correspondence_plot_from_kpts(image1, image2, kpts1, kpts2, confidence=None):
    # Create new pixel maps from the keypoints
    h, w, _ = image1.shape
    pixel_map_12 = np.full((h, w, 2), -1, dtype=np.int32)
    pixel_map_21 = np.full((h, w, 2), -1, dtype=np.int32)
    pixel_map_12[kpts1[:, 1], kpts1[:, 0]] = kpts2
    pixel_map_21[kpts2[:, 1], kpts2[:, 0]] = kpts1

    create_interactive_correspondence_plot(image1, image2, pixel_map_12, pixel_map_21, confidence)

def create_interactive_correspondence_plot(image1, image2, pixel_map_12, pixel_map_21, confidence=None):
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(121)
    ax1.imshow(image1)
    ax1.set_title('Image 1')
    ax1.axis('off')

    ax2 = fig.add_subplot(122)
    ax2.imshow(image2)
    ax2.set_title('Image 2')
    ax2.axis('off')

    circle_ax1 = Circle((0, 0), radius=1, edgecolor='red', facecolor='none', linewidth=2)
    circle_ax2 = Circle((0, 0), radius=1, edgecolor='red', facecolor='none', linewidth=2)

    circle_ax1.set_visible(False)
    circle_ax2.set_visible(False)
    textbox_ax1 = TextArea("Test")
    textbox_ax2 = TextArea("Test")
    ax1.add_patch(circle_ax1)
    ax2.add_patch(circle_ax2)
    ab_1 = AnnotationBbox(textbox_ax1, (0,0),
                xybox=(30,30),
                xycoords='data',
                boxcoords=("offset points"),
                box_alignment=(0., 0.5),
                arrowprops=dict(arrowstyle="->"))
    ab_1.set_visible(False)
    ax1.add_artist(ab_1)
    ab_2 = AnnotationBbox(textbox_ax2, (0,0),
                xybox=(30,30),
                xycoords='data',
                boxcoords=("offset points"),
                box_alignment=(0., 0.5),
                arrowprops=dict(arrowstyle="->"))
    ab_2.set_visible(False)
    ax2.add_artist(ab_2)

    def on_click(event):
        if event.inaxes == ax1:
            x, y = int(event.xdata), int(event.ydata)
            circle_1_pos = (x, y)
            circle_2_pos = pixel_map_12[y, x]
        elif event.inaxes == ax2:
            x, y = int(event.xdata), int(event.ydata)
            circle_1_pos = pixel_map_21[y, x]
            circle_2_pos = (x, y)
        else:
            circle_ax1.set_visible(False)
            circle_ax2.set_visible(False)
            return
        circle_ax1.set_center((circle_1_pos[0], circle_1_pos[1]))
        circle_ax2.set_center((circle_2_pos[0], circle_2_pos[1]))
        circle_ax1.set_visible(True)
        circle_ax2.set_visible(True)
        if circle_1_pos[0] < 0 or circle_1_pos[1] < 0:
            textbox_ax1.set_text('No bijective match found')
            ab_1.xy = (image1.shape[1]/2, image1.shape[0]/2)
        else:
            if confidence is not None:
                conf_value = confidence[0, 0, circle_1_pos[1], circle_1_pos[0]]
                textbox_ax1.set_text(f'({circle_1_pos[0]}, {circle_1_pos[1]})\nConfidence: {conf_value:.2f}')
            else:
                textbox_ax1.set_text(f'({circle_1_pos[0]}, {circle_1_pos[1]})')
            ab_1.xy = (circle_1_pos[0], circle_1_pos[1])
        if circle_2_pos[0] < 0 or circle_2_pos[1] < 0:
            textbox_ax2.set_text('No bijective match found')
            ab_2.xy = (image2.shape[1]/2, image2.shape[0]/2)
        else:
            if confidence is not None:
                conf_value = confidence[0, 1, circle_2_pos[1], circle_2_pos[0]]
                textbox_ax2.set_text(f'({circle_2_pos[0]}, {circle_2_pos[1]})\nConfidence: {conf_value:.2f}')
            else:
                textbox_ax2.set_text(f'({circle_2_pos[0]}, {circle_2_pos[1]})')
            ab_2.xy = (circle_2_pos[0], circle_2_pos[1])
        ab_1.set_visible(True)
        ab_2.set_visible(True)

        fig.canvas.draw()
        

    fig.canvas.mpl_connect('button_press_event', on_click)
    # mplcursors.cursor(hover=True)
    plt.show()

def plot_correspondences(img1, img2, kpts1, kpts2, draw_lines=True, filter_invalid=True):
    # write code that displays the projected points on the second image using matplotlib
    plt.figure()
    ax1 = plt.subplot(1, 2, 1)
    plt.imshow(img1)
    # show only the points that are within the image bounds
    h, w, _ = img1.shape
    if filter_invalid:
        kpts1 = kpts1[(kpts1[:, 0] >= 0) & (kpts1[:, 0] < w) & (kpts1[:, 1] >= 0) & (kpts1[:, 1] < h)]
    projected_points_plot = kpts1.reshape(-1,2)
    plt.scatter(projected_points_plot[:, 0], projected_points_plot[:, 1], s=1, c='r')
    ax2 = plt.subplot(1, 2, 2)
    plt.imshow(img2)
    # show only the points that are within the image bounds
    h, w, _ = img2.shape
    if filter_invalid:
        kpts2 = kpts2[(kpts2[:, 0] >= 0) & (kpts2[:, 0] < w) & (kpts2[:, 1] >= 0) & (kpts2[:, 1] < h)]
    projected_points_plot = kpts2.reshape(-1,2)
    plt.scatter(projected_points_plot[:, 0], projected_points_plot[:, 1], s=1, c='r')

    if draw_lines:
        for i in range(min(len(kpts1), len(kpts2))):
            con = matplotlib.patches.ConnectionPatch(xyA=(kpts2[i, 0], kpts2[i, 1]), xyB=(kpts1[i, 0], kpts1[i, 1]),
                                                   coordsA="data", coordsB="data",
                                                   axesA=ax2, axesB=ax1, color="red", linewidth=0.5)
            ax2.add_artist(con)
    plt.show()

def make_matching_figure(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None):
    # Taken from XoFTR repository
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):   # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                            (fkpts0[i, 1], fkpts1[i, 1]),
                                            transform=fig.transFigure, c=color[i], linewidth=1)
                                        for i in range(len(mkpts0))]
        
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig

if __name__ == "__main__":
    # Example usage with dummy data
    images = load_and_preprocess_images(["/home/landson/RGBT-Scenes/Building/rgb/test/img_249.jpg", "/home/landson/RGBT-Scenes/Building/rgb/test/img_257.jpg"])
    pixel_map_12 = np.dstack((np.random.randint(0, images[0].shape[2], (images[0].shape[1], images[0].shape[2])), np.random.randint(0, images[0].shape[1], (images[0].shape[1], images[0].shape[2]))))
    pixel_map_21 = np.dstack((np.random.randint(0, images[1].shape[2], (images[1].shape[1], images[1].shape[2])), np.random.randint(0, images[1].shape[1], (images[1].shape[1], images[1].shape[2]))))

    create_interactive_correspondence_plot(images[0].permute(1, 2, 0), images[1].permute(1, 2, 0), pixel_map_12, pixel_map_21)