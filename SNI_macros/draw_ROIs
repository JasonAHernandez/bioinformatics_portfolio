import os
import cv2
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Polygon

class InteractiveROIDrawer:
    def __init__(self, bf_folder, movie_folder, output_folder, index_formula):
        self.bf_folder = bf_folder
        self.movie_folder = movie_folder
        self.output_folder = output_folder
        self.index_formula = index_formula
        os.makedirs(self.output_folder, exist_ok=True)

        self.roi_points = []
        self.mask = np.zeros((370, 370), dtype=np.uint8)
        self.current_image_name = None

        self.img_bf = None
        self.img_movie = None
        self.raw_movie_frame = None

        self.fig = None
        self.bf_ax = None
        self.movie_ax = None
        self.slider_contrast = None
        self.edit_mode = False
        self.selected_point_idx = None  # Track which point is being dragged

    def get_matching_movie(self, bf_name):
        base_index = int(bf_name.split('_')[-1].split('.')[0])
        try:
            movie_index = eval(self.index_formula, {"index": base_index})
            movie_name = bf_name.replace(f"{base_index:03d}", f"{movie_index:03d}").replace('.tif', '_cleaned.tif')
            return movie_name
        except Exception as e:
            print(f"Error evaluating index formula '{self.index_formula}' with index {base_index}: {e}")
            return None

    def load_bf_images(self):
        return [f for f in os.listdir(self.bf_folder)
                if f.endswith('.tif') and tifffile.imread(os.path.join(self.bf_folder, f)).shape[:2] == (370, 370)]

    def draw_mask_from_points(self):
        self.mask = np.zeros((370, 370), dtype=np.uint8)
        if len(self.roi_points) >= 3:
            cv2.fillPoly(self.mask, [np.array(self.roi_points, dtype=np.int32)], color=255)

    def update_display(self):
        for ax in (self.bf_ax, self.movie_ax):
            ax.clear()

        self.bf_ax.imshow(self.img_bf, cmap='gray')
        self.movie_ax.imshow(self.img_movie, cmap='gray')

        if len(self.roi_points) >= 2:
            for ax in (self.bf_ax, self.movie_ax):
                poly = Polygon(self.roi_points, closed=True, edgecolor='lime', fill=False, linewidth=1.5)
                ax.add_patch(poly)
        for ax in (self.bf_ax, self.movie_ax):
            ax.plot(*zip(*self.roi_points), marker='o', color='lime')

        self.fig.canvas.draw_idle()

    def onclick(self, event):
        if event.inaxes not in [self.bf_ax, self.movie_ax] or event.xdata is None or event.ydata is None:
            return

        x, y = int(event.xdata), int(event.ydata)

        if self.edit_mode:
            # Try to select a nearby point
            for i, (px, py) in enumerate(self.roi_points):
                if np.hypot(px - x, py - y) < 10:  # 10-pixel threshold
                    self.selected_point_idx = i
                    break
        else:
            self.roi_points.append((x, y))
            self.draw_mask_from_points()
            self.update_display()

    def onscroll(self, event):
        base_scale = 1.2
        ax = event.inaxes
        if ax not in [self.bf_ax, self.movie_ax] or event.xdata is None or event.ydata is None:
            return

        xdata, ydata = event.xdata, event.ydata
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xcenter = xdata
        ycenter = ydata

        scale_factor = 1 / base_scale if event.button == 'up' else base_scale

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xcenter) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ycenter) / (cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([xcenter - new_width * (1 - relx), xcenter + new_width * relx])
        ax.set_ylim([ycenter - new_height * (1 - rely), ycenter + new_height * rely])
        self.fig.canvas.draw_idle()

    def onrelease(self, event):
        self.selected_point_idx = None

    def onmotion(self, event):
        if not self.edit_mode or self.selected_point_idx is None:
            return
        if event.inaxes not in [self.bf_ax, self.movie_ax] or event.xdata is None or event.ydata is None:
            return

        x, y = int(event.xdata), int(event.ydata)
        self.roi_points[self.selected_point_idx] = (x, y)
        self.draw_mask_from_points()
        self.update_display()

    def apply_contrast_enhancement(self, val):
        if self.raw_movie_frame is None:
            return

        # Normalize slider value: 1.0 = no change, <1.0 = less contrast, >1.0 = more contrast
        factor = float(val)

        # Convert to float for safe manipulation
        img = self.raw_movie_frame.astype(np.float32)

        # Center contrast around mean
        mean = np.mean(img)
        enhanced = (img - mean) * factor + mean

        # Clip and convert back to uint8
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

        self.img_movie = enhanced
        self.update_display()

    def undo_point(self, event):
        if self.roi_points:
            self.roi_points.pop()
            self.draw_mask_from_points()
            self.update_display()

    def save_mask(self, event=None):
        save_path = os.path.join(self.output_folder, self.current_image_name)
        tifffile.imwrite(save_path, self.mask)
        print(f"Mask saved to: {save_path}")
        plt.close(self.fig)

    def skip_image(self, event=None):
        print(f"Skipped: {self.current_image_name}")
        plt.close(self.fig)

    def toggle_edit_mode(self, event=None):
        self.edit_mode = not self.edit_mode
        mode = "Edit" if self.edit_mode else "Draw"
        print(f"Mode switched to: {mode}")

    def show_image_pair(self, bf_path, movie_path):
        self.img_bf = tifffile.imread(bf_path)
        self.raw_movie_frame = tifffile.imread(movie_path)[0]
        self.img_movie = self.raw_movie_frame.copy()
        self.mask = np.zeros((370, 370), dtype=np.uint8)
        self.roi_points = []

        self.fig, (self.bf_ax, self.movie_ax) = plt.subplots(1, 2, figsize=(10, 5))
        self.bf_ax.set_title("Brightfield")
        self.movie_ax.set_title("Movie Frame 1")

        self.fig.subplots_adjust(bottom=0.3)

        # Buttons
        ax_save = plt.axes([0.1, 0.05, 0.15, 0.075])
        ax_skip = plt.axes([0.3, 0.05, 0.15, 0.075])
        ax_undo = plt.axes([0.5, 0.05, 0.15, 0.075])
        btn_save = Button(ax_save, 'Save Mask')
        btn_skip = Button(ax_skip, 'Skip')
        btn_undo = Button(ax_undo, 'Undo Point')
        btn_save.on_clicked(self.save_mask)
        btn_skip.on_clicked(self.skip_image)
        btn_undo.on_clicked(self.undo_point)
        ax_toggle = plt.axes([0.7, 0.15, 0.25, 0.075])
        btn_toggle = Button(ax_toggle, 'Toggle Edit Mode')
        btn_toggle.on_clicked(self.toggle_edit_mode)

        # Contrast slider
        contrast_ax = plt.axes([0.7, 0.08, 0.25, 0.03])
        self.slider_contrast = Slider(
            ax=contrast_ax,
            label='Contrast Scale',
            valmin=-1.0,  # lower = less contrast
            valmax=5.0,  # higher = more contrast
            valinit=1.0,  # 1.0 = original image
            valstep=0.05
        )

        self.slider_contrast.on_changed(self.apply_contrast_enhancement)

        # Events
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        self.fig.canvas.mpl_connect('motion_notify_event', self.onmotion)
        self.fig.canvas.mpl_connect('button_release_event', self.onrelease)

        self.update_display()
        plt.show()

    def run(self):
        bf_images = self.load_bf_images()
        for bf_name in bf_images:
            self.current_image_name = bf_name
            movie_name = self.get_matching_movie(bf_name)
            bf_path = os.path.join(self.bf_folder, bf_name)
            movie_path = os.path.join(self.movie_folder, movie_name)
            if os.path.exists(movie_path):
                print(f"Showing: {bf_name} + {movie_name}")
                self.show_image_pair(bf_path, movie_path)
            else:
                print(f"Movie not found for: {bf_name}")
