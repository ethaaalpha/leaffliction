from pathlib import Path
import random
from PIL import Image as Im
from PIL import ImageEnhance, ImageFilter
import os, sys

class ImageAugmentation:
    def __init__(self):
        self.results = {}
        self.original_name = ""

    def generate(self, image_path: str):
        self.original_name = Path(image_path).stem
        self.original_ext = Path(image_path).suffix[1:]

        functions = [self.rotate, self.transpose, self.blur, self.contrast, self.brightness, self.scale, self.projective]
        
        with Im.open(image_path) as image:
            self.results = { func.__name__: func(image) for func in functions }

    def export(self, dest_path: str):
        os.makedirs(dest_path, exist_ok=True)

        if (os.path.isdir(dest_path)):
            dest_path = Path(dest_path)

            for k, v in self.results.items():
                v: Im.Image
                image_path = dest_path / f"{self.original_name}_{k}.{self.original_ext}"
                v.save(image_path.resolve(), self.original_ext)

    @classmethod
    def rotate(cls, image: Im.Image) -> Im.Image:
        return image.rotate(random.randrange(10, 40))
    
    @classmethod
    def transpose(cls, image: Im.Image) -> Im.Image:
        return image.transpose(random.choice(Im.Transpose.FLIP_TOP_BOTTOM))
    
    @classmethod
    def blur(cls, image: Im.Image) -> Im.Image:
        return image.filter(ImageFilter.GaussianBlur(random.randrange(5, 30)))
    
    @classmethod
    def contrast(cls, image: Im.Image) -> Im.Image:
        return ImageEnhance.Contrast(image).enhance(random.uniform(1.9, 4.0))
    
    @classmethod
    def brightness(cls, image: Im.Image) -> Im.Image:
        return ImageEnhance.Brightness(image).enhance(random.uniform(0.3, 0.7))
    
    @classmethod
    def scale(cls, image: Im.Image) -> Im.Image:
        zoom = random.uniform(0.1, 0.4)
        w, h = image.width, image.height

        zm = image.crop((w * zoom, h * zoom, w - w * zoom, h - h * zoom))
        return zm.resize((w, h))
    
    @classmethod
    def projective(cls, image: Im.Image) -> Im.Image:
        a = 1 + random.uniform(-0.25, 0.25)  # scale X
        b = random.uniform(-0.15, 0.15)      # shear X
        c = random.uniform(-50, 50)          # translate X
        d = random.uniform(-0.15, 0.15)      # shear Y
        e = 1 + random.uniform(-0.25, 0.25)  # scale Y
        f = random.uniform(-50, 50)          # translate Y

        w, h = image.width, image.height

        return image.transform((w, h), Im.AFFINE, (a, b, c, d, e, f))

def main():
    if len(sys.argv) == 2:
        imgA = ImageAugmentation()

        imgA.generate(sys.argv[1])
        imgA.export("./augmented_directory")

if __name__ == "__main__":
    main()