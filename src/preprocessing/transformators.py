import cv2
import pandas
import numpy as np
from matplotlib.figure import Figure
from plantcv import plantcv as pcv

class Transformators:
    def __init__(self, image_path: str):
        self._image = self.load_image(image_path)

    def grayscale(self) -> np.ndarray:
        return pcv.rgb2gray_hsv(self._image, 's')

    def pseudocolored(self) -> Figure:
        gray = self.grayscale()
        mask = self.mask()
        height, width = gray.shape
        dpi = 100

        fig = pcv.visualize.pseudocolor(gray_img=gray, mask=mask, background="image", 
            axes=False, colorbar=False, cmap='viridis')
        # set dims
        fig.set_dpi(dpi)
        fig.set_figheight(height / dpi)
        fig.set_figwidth(width / dpi)
        # remove borders
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.tight_layout(pad=0)

        return fig
    
    def shape_size(self) -> np.ndarray:
        mask = self.mask()

        return pcv.analyze.size(self._image, mask)

    def pseudolandmarks(self) -> np.ndarray:
        size = 4
        mask = self.mask()
        top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(img=self._image, mask=mask)

        # as in plantcv
        cpy = np.copy(self._image)
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

    def mask(self) -> np.ndarray:
        gray = self.grayscale()
        return pcv.threshold.binary(gray, threshold=80, object_type='light')

    def color_histogram(self) -> Figure | pandas.DataFrame:
        mask = self.mask()

        _, hist_data = pcv.visualize.histogram(self._image, mask, hist_data=True)
        return hist_data

    @classmethod
    def load_image(cls, image_path: str) -> np.ndarray:
        img_array, _, _ = pcv.readimage(image_path)
        return img_array

    @classmethod
    def export_image(cls, image: np.ndarray | Figure, image_path: str):
        pcv.print_image(image, image_path)
