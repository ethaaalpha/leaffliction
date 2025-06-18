import math
import os
from pathlib import Path
import shutil
from src.preprocessing.img_augmentation import ImageAugmentation as IAug
from src.preprocessing.loader import Loader
from PIL import Image as Im
import random

random.seed(42)

# preference order for augmentations
AUGMENTATIONS = [IAug.rotate, IAug.contrast, IAug.brightness, IAug.projective, IAug.blur, IAug.scale]

def determine_missing(tab: dict[str, int], target_per_class: int) -> dict[str, int]:
    missing = {}

    for k, v in tab.items():
        missing[k] = target_per_class - v
    return missing

def determine_augmentation_per_images(count: dict[str, int], missing: dict[str, int]) -> dict[str, float]:
    augm = {}

    for k, v in missing.items():
        augm[k] = v / count[k] 
    return augm

def copy_original_images(tab: dict[str, list[str]], result_directory: str):
    for _class, files in tab.items():
        new_directory = os.path.join(result_directory, _class)
        os.makedirs(new_directory, exist_ok=True)

        for file in files:
            dest_path = os.path.join(new_directory, os.path.basename(file))
            shutil.copyfile(file, dest_path)


def generate_image(result_directory: str, files: list[str], augm_per_image: float, max_imgs: int):
    count = 0
    random.shuffle(files)

    for img_path in files:
        with Im.open(img_path) as img:
            for i in range(math.ceil(augm_per_image)):
                if count >= max_imgs:
                    break
                else:
                    image_augmented = AUGMENTATIONS[i](img)
                    augmentation_name = AUGMENTATIONS[i].__name__
                    name, ext = os.path.splitext(os.path.basename(img.filename))

                    image_augmented.save(os.path.join(result_directory, f"{name}_{augmentation_name}{ext}"))
                    count += 1
                    print(f"{count}/{max_imgs} generated!", end='\r', flush=True)

def generate_balanced_dataset(tab: dict[str, list[str]], count: dict[str, int], result_directory: str):
    target_per_class = min(count.values()) * (len(AUGMENTATIONS) + 1) # adding the original ones
    missing = determine_missing(count, target_per_class)
    per_image = determine_augmentation_per_images(count, missing)

    for _class, files in tab.items():
        print(f"Generating augmented images for {_class}..")
        copy_original_images(tab, result_directory)
        generate_image(
            os.path.join(result_directory, _class),
            files,
            per_image[_class], 
            missing[_class])

def split_dataset(dataset_path: str, split: float):
    directory_name = os.path.basename(dataset_path)

    data = Loader(dataset_path).parse()

    for _class, files in data.items():
        training_path = os.path.join(f"{directory_name}_training", _class)
        validation_path = os.path.join(f"{directory_name}_validation", _class)

        os.makedirs(training_path, exist_ok=True)
        os.makedirs(validation_path, exist_ok=True)
    
        split_index = int(len(files) * split)
        for i, file in enumerate(files):
            name = os.path.basename(file)

            if i < split_index:
                shutil.copyfile(file, os.path.join(validation_path, name))
            else:
                shutil.copyfile(file, os.path.join(training_path, name))
