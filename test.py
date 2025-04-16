import numpy as np
from plantcv import plantcv as pcv
from PIL import Image

from src.preprocessing.augmentation import ImageAugmentation


augm = ImageAugmentation()

augm.generate("base.png")

augm.export("bis")
augm.export("test/")