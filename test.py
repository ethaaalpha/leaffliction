import numpy as np
from plantcv import plantcv as pcv
from PIL import Image

from src.preprocessing.transformation import ImageTransformation
from src.preprocessing.augmentation import ImageAugmentation


augm = ImageAugmentation()

augm.generate("image.png")
augm.export("augm")

trans = ImageTransformation()

trans.generate("image.png")
trans.export("transf")