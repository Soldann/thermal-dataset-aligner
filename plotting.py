import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vggt.utils.load_fn import load_and_preprocess_images
import mplcursors
from matplotlib.patches import Circle
from matplotlib.offsetbox import AnnotationBbox, TextArea

def create_interactive_plot(image1, image2, pixel_map_12, pixel_map_21):
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
        textbox_ax1.set_text(f'({circle_1_pos[0]}, {circle_1_pos[1]})')
        textbox_ax2.set_text(f'({circle_2_pos[0]}, {circle_2_pos[1]})')
        ab_1.xy = (circle_1_pos[0], circle_1_pos[1])
        ab_2.xy = (circle_2_pos[0], circle_2_pos[1])
        ab_1.set_visible(True)
        ab_2.set_visible(True)

        fig.canvas.draw()
        

    fig.canvas.mpl_connect('button_press_event', on_click)
    # mplcursors.cursor(hover=True)
    plt.show()

if __name__ == "__main__":
    # Example usage with dummy data
    images = load_and_preprocess_images(["/home/landson/RGBT-Scenes/Building/rgb/test/img_249.jpg", "/home/landson/RGBT-Scenes/Building/rgb/test/img_257.jpg"])
    pixel_map_12 = np.dstack((np.random.randint(0, images[0].shape[2], (images[0].shape[1], images[0].shape[2])), np.random.randint(0, images[0].shape[1], (images[0].shape[1], images[0].shape[2]))))
    pixel_map_21 = np.dstack((np.random.randint(0, images[1].shape[2], (images[1].shape[1], images[1].shape[2])), np.random.randint(0, images[1].shape[1], (images[1].shape[1], images[1].shape[2]))))



    create_interactive_plot(images[0].permute(1, 2, 0), images[1].permute(1, 2, 0), pixel_map_12, pixel_map_21)