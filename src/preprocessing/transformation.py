import os
import cv2
import numpy as np
from pathlib import Path
from plantcv import plantcv as pcv

class ImageTransformation:
    def __init__(self):
        self.results = {}
        self.original_name = ""
        self.original_ext = ""

    def generate(self, image_path: str):
        self.original_name = Path(image_path).stem
        self.original_ext = Path(image_path).suffix[1:]

        functions = [self.grayscale, self.pseudocolored, self.shape_size, self.pseudolandmarks, self.mask, self.color_histogram]
        image, _, _ = pcv.readimage(image_path)

        self.results = { func.__name__: func(image) for func in functions }
    
    def export(self, dest_path: str):
        os.makedirs(dest_path, exist_ok=True)

        if (os.path.isdir(dest_path)):
            dest_path = Path(dest_path)

            for k, v in self.results.items():
                pcv.print_image(v, dest_path / f"{self.original_name}_{k}.{self.original_ext}")

    @classmethod
    def grayscale(cls, image: np.ndarray):
        return pcv.rgb2gray_hsv(image, 's')

    @classmethod
    def pseudocolored(cls, image: np.ndarray):
        gray = cls.grayscale(image)
        mask = cls.mask(image)

        return pcv.visualize.pseudocolor(gray_img=gray, mask=mask, cmap='viridis',
            colorbar=False, axes=False, background="white")
    
    @classmethod
    def shape_size(cls, image: np.ndarray):
        mask = cls.mask(image)

        return pcv.analyze.size(image, mask)

    @classmethod
    def pseudolandmarks(cls, image: np.ndarray):
        size = 12
        mask = cls.mask(image)
        top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(img=image, mask=mask)

        # as in plantcv
        cpy = np.copy(image)
        for i in top:
            x = i[0, 0]
            y = i[0, 1]
            cv2.circle(cpy, (int(x), int(y)), size, (255, 0, 0), -1)
        for i in bottom:
            x = i[0, 0]
            y = i[0, 1]
            cv2.circle(cpy, (int(x), int(y)), size, (255, 0, 255), -1)
        for i in center_v:
            x = i[0, 0]
            y = i[0, 1]
            cv2.circle(cpy, (int(x), int(y)), size, (0, 79, 255), -1)

        return cpy

    @classmethod
    def mask(cls, image: np.ndarray):
        gray = cls.grayscale(image)
        return pcv.threshold.binary(gray, threshold=80, object_type='light')

    @classmethod
    def color_histogram(cls, image: np.ndarray):
        mask = cls.mask(image)

        return pcv.visualize.histogram(image, mask)
