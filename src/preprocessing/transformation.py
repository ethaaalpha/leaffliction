# import os
import numpy as np
from plantcv import plantcv as pcv
# from pathlib import Path
# import os.path as path

class ImageTransformation:
    def __init__(self):
        self.results = {}
        self.original_name = ""

    def generate(self, image_path: str):
        image, path, filename = pcv.readimage(image_path)
    
    # def export(self, dest_path: str):
    #     os.makedirs(dest_path, exist_ok=True)

    #     if (path.isdir(dest_path)):
    #         dest_path = Path(dest_path)

    #         for k, v in self.results.items():
    #             v: Im.Image
    #             image_path = dest_path / f"{self.original_name}_{k}.{self.original_ext}"
    #             v.save(image_path.resolve(), self.original_ext)

    @staticmethod
    def glaussian_blur(image: np.ndarray):
        return pcv.gaussian_blur(image, ksize=(51, 51), sigma_x=0, sigma_y=None)
    
    @staticmethod
    def mask(image: np.ndarray):
        gray = pcv.rgb2gray_lab(image, channel='b')
        mask = pcv.threshold.binary(gray, threshold=120, object_type='light')
        masked_img = pcv.apply_mask(image, mask=mask, mask_color='black')
        mask = pcv.fill(mask, size=200)

        # print(masked_img)
        return masked_img


it = ImageTransformation()

image, path_, filename = pcv.readimage("image.png")

pcv.plot_image(image)
pcv.plot_image(it.mask(image))