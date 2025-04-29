from pathlib import Path
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
        return image.rotate(30)
    
    @classmethod
    def transpose(cls, image: Im.Image) -> Im.Image:
        return image.transpose(Im.Transpose.FLIP_TOP_BOTTOM)
    
    @classmethod
    def blur(cls, image: Im.Image) -> Im.Image:
        return image.filter(ImageFilter.GaussianBlur(15))
    
    @classmethod
    def contrast(cls, image: Im.Image) -> Im.Image:
        return ImageEnhance.Contrast(image).enhance(2.6)
    
    @classmethod
    def brightness(cls, image: Im.Image) -> Im.Image:
        return ImageEnhance.Brightness(image).enhance(0.6)
    
    @classmethod
    def scale(cls, image: Im.Image) -> Im.Image:
        zoom = 0.2
        w, h = image.width, image.height

        zm = image.crop((w * zoom, h * zoom, w - w * zoom, h - h * zoom))
        return zm.resize((w, h))
    
    @classmethod
    def projective(cls, image: Im.Image) -> Im.Image:
        t_data = [1.25, 0.1, -110, 0.11, 1.25, -110]
        w, h = image.width, image.height

        return image.transform((w, h), Im.AFFINE, t_data)

def main():
    if len(sys.argv) == 2:
        imgA = ImageAugmentation()

        imgA.generate(sys.argv[1])
        imgA.export("./augmented_directory")

if __name__ == "__main__":
    main()