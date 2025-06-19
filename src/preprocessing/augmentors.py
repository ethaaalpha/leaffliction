from PIL import Image as Im
from PIL import ImageEnhance, ImageFilter
import random

class Augmentators:
    def __init__(self, image_path: str):
        self._image = self.load_image(image_path)

    def rotate(self) -> Im.Image:
        return self._image.rotate(random.randrange(-40, 40))
    
    def transpose(self) -> Im.Image:
        return self._image.transpose(random.choice(list(Im.Transpose)))
    
    def blur(self) -> Im.Image:
        return self._image.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 4)))
    
    def contrast(self) -> Im.Image:
        return ImageEnhance.Contrast(self._image).enhance(random.uniform(1.5, 2.0))
    
    def brightness(self) -> Im.Image:
        return ImageEnhance.Brightness(self._image).enhance(random.uniform(0.5, 1.5))
    
    def scale(self) -> Im.Image:
        zoom = random.uniform(0.1, 0.4)
        w, h = self._image.width, self._image.height

        zm = self._image.crop((w * zoom, h * zoom, w - w * zoom, h - h * zoom))
        return zm.resize((w, h))
    
    def projective(self) -> Im.Image:
        a = 1 + random.uniform(-0.25, 0.25)
        b = random.uniform(-0.15, 0.15)
        c = random.uniform(-50, 50)
        d = random.uniform(-0.15, 0.15)
        e = 1 + random.uniform(-0.25, 0.25)
        f = random.uniform(-50, 50)

        w, h = self._image.width, self._image.height

        return self._image.transform((w, h), Im.AFFINE, (a, b, c, d, e, f))

    @classmethod
    def load_image(cls, image_path: str) -> Im.Image:
        with Im.open(image_path) as img:
            return img.copy()

    @classmethod
    def export_image(cls, image: Im.Image, image_path: str):
        image.save(image_path)
