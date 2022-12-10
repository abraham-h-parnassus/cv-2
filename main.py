import os.path
from tkinter import *
from tkinter import filedialog as fd, Image
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from affine import affine_transform, normalize, _do_transform
from utils import show_image

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


class Application:

    def __init__(self, root):
        self.root = root
        self.frame = ttk.Frame(root, padding=10)
        self.params_section = ttk.Labelframe(self.frame, text='Affine Transformation Parameters')
        self.reference_image_section = ttk.Labelframe(self.frame, text='Reference Image')
        self.transformed_image_section = ttk.Labelframe(self.frame, text='Transformed Image')
        self.actions_section = ttk.Labelframe(self.frame, text='')
        self.images_section = ttk.Labelframe(self.frame, text='Result Image')

        self.params = [
            ttk.Entry(self.params_section, width=3),
            ttk.Entry(self.params_section, width=3),
            ttk.Entry(self.params_section, width=3),
            ttk.Entry(self.params_section, width=3),
            ttk.Entry(self.params_section, width=3),
            ttk.Entry(self.params_section, width=3)
        ]
        for param in self.params:
            param.insert(0, '0')

        self.reference_image_btn = ttk.Button(self.reference_image_section, text="Choose reference image",
                                              command=self.select_ref_image)
        self.reference_image_label = ttk.Label(self.reference_image_section, text="n/a")
        self.reference_image_file = None
        self.reference_image_view = Label(self.images_section, borderwidth=0)
        self.reference_image_np = None

        self.transformed_image_btn = ttk.Button(self.transformed_image_section, text="Choose transformed image",
                                                command=self.select_trans_image)
        self.transformed_image_label = ttk.Label(self.transformed_image_section, text="n/a")
        self.transformed_image_file = None
        self.transformed_image_view = Label(self.images_section, borderwidth=0)
        self.transformed_image_np = None
        self.result_image_view = Label(self.images_section)

        self.transform_button = ttk.Button(self.actions_section, text="Transform",
                                           command=self.transform)

        self.normalize_button = ttk.Button(self.actions_section, text="Normalize",
                                           command=self.normalize)
        self.reference_points = []
        self.transformed_points = []
        self.reference_image_view.bind("<Button-1>",
                                       self.on_mouse_click(self.reference_image_view, self.reference_points,
                                                           lambda: self.reference_image_np))
        self.transformed_image_view.bind("<Button-1>",
                                         self.on_mouse_click(self.transformed_image_view, self.transformed_points,
                                                             lambda: self.transformed_image_np))

    def on_mouse_click(self, label, points, image):
        def click_handler(event):
            if len(points) == 3:
                points.clear()
            points.append((event.x, event.y))
            self._show_image(label, image(), points)

        return click_handler

    def show(self):
        self.frame.grid()
        self.params_section.grid(column=0, row=0)
        self.reference_image_section.grid(column=0, row=1)
        self.transformed_image_section.grid(column=0, row=2)
        self.actions_section.grid(column=0, row=3)
        self.images_section.grid(column=1, row=0, rowspan=3)

        self.params[0].grid(column=0, row=0)
        self.params[1].grid(column=1, row=0)
        self.params[2].grid(column=2, row=0)
        self.params[3].grid(column=0, row=1)
        self.params[4].grid(column=1, row=1)
        self.params[5].grid(column=2, row=1)

        self.reference_image_btn.grid(column=0, row=0)
        self.reference_image_label.grid(column=0, row=1)

        self.transformed_image_btn.grid(column=0, row=0)
        self.transformed_image_label.grid(column=0, row=1)

        self.reference_image_view.grid(column=0, row=0)
        self.transformed_image_view.grid(column=0, row=1)
        self.result_image_view.grid(column=0, row=2)

        self.transform_button.grid(column=0, row=0)
        self.normalize_button.grid(column=1, row=0)

        self.root.mainloop()

    def transform(self):
        params_ = [float(entry.get()) for entry in self.params]
        self.transformed_image_np = affine_transform(self.reference_image_file, params_)
        self._show_image(self.transformed_image_view, self.transformed_image_np)

        T = np.array([
            [params_[0], params_[1], params_[2]],
            [params_[3], params_[4], params_[5]],
            [0, 0, 1]
        ])
        RT = np.linalg.inv(T)
        result = _do_transform(self.transformed_image_np, [RT[0][0], RT[0][1], RT[0][2], RT[1, 0], RT[1, 1], RT[1, 2]])
        self._show_image(self.result_image_view, result)

    def normalize(self):
        result = normalize(
            self.reference_points,
            self.transformed_image_np,
            self.transformed_points
        )
        show_image(result)
        self._show_image(self.result_image_view, result)

    def select_ref_image(self):
        path = fd.askopenfile()
        base_name = os.path.basename(path.name)
        self.reference_image_label.configure(text=base_name)
        self.reference_image_file = path.name

        image = cv2.imread(path.name)
        self.reference_image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self._show_image(self.reference_image_view, self.reference_image_np)

    def select_trans_image(self):
        path = fd.askopenfile()
        base_name = os.path.basename(path.name)
        self.transformed_image_label.configure(text=base_name)
        self.transformed_image_file = path.name

        image = cv2.imread(path.name)
        self.transformed_image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self._show_image(self.transformed_image_view, self.transformed_image_np)

    def _show_image(self, panel: Label, image, points=None):
        temp_image = image
        if points:
            temp_image = np.copy(image)
            for i, point in enumerate(points):
                cv2.circle(temp_image, center=(point[0], point[1]), color=colors[i], thickness=2, radius=4)
        photo_image = ImageTk.PhotoImage(image=Image.fromarray(temp_image, 'RGB'))
        panel.photo = photo_image
        panel.configure(image=photo_image)
        self.root.update()


def show_ui():
    root = Tk()
    Application(root).show()
    root.mainloop()


if __name__ == '__main__':
    show_ui()
