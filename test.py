import numpy as np
from plantcv import plantcv as pcv
from PIL import Image

from src.augmentation import generate_balanced_dataset, split_dataset
from src.preprocessing.img_transformation import ImageTransformation
from src.preprocessing.img_augmentation import ImageAugmentation
from src.preprocessing.loader import Loader

# augm = ImageAugmentation()

# augm.generate("image.png")
# augm.export("augm")

# trans = ImageTransformation()

# trans.generate("image.png")
# trans.export("transf")

loader = Loader("images")
# tab = Loader().count("images")
# missing = determine_missing(tab)
# transf_nb = determine_transformation_per_images(tab, missing)

# print(tab)
# print(missing)
# print(transf_nb)

generate_balanced_dataset(loader.parse(), loader.count(), "augmented")
split_dataset("augmented", 0.1)