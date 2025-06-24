import math
import os
import shutil
import random
from PIL import Image as Im
from argparse import ArgumentParser, ArgumentTypeError
from matplotlib import pyplot as plt
from src.preprocessing.augmentors import Augmentators as Augs
from src.preprocessing.loader import Loader
from src.utils import log, log_dynamic
from os.path import join

random.seed(42)

# preference order for augmentations
AUGMENTATIONS = [Augs.rotate, Augs.contrast, Augs.brightness,
                 Augs.projective, Augs.blur, Augs.scale]


def determine_missing(tab: dict[str, int], t_per_class: int) -> dict[str, int]:
    missing = {}

    for k, v in tab.items():
        missing[k] = t_per_class - v
    return missing


def determine_augmentation_per_images(count, missing) -> dict[str, float]:
    augm = {}

    for k, v in missing.items():
        augm[k] = v / count[k]
    return augm


def plot_images(images: dict[str, Im.Image]):
    _, axs = plt.subplots(2, 3)

    for ax, (name, img) in zip(axs.flatten(), images.items()):
        ax.imshow(img)
        ax.set_title(name)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def generate_augmentation(dir, files, per_image, max_imgs, display=False):
    count = 0
    random.shuffle(files)
    to_display = {}

    for img_path in files:
        augmentor = Augs(img_path)

        for i in range(math.ceil(per_image)):
            if count >= max_imgs:
                break
            else:
                image_augmented = AUGMENTATIONS[i](augmentor)
                augmentation_name = AUGMENTATIONS[i].__name__
                name, ext = Loader.get_name_ext(img_path)

                log_dynamic(
                    f"working on: {name.ljust(42)} ({count}/{max_imgs})")

                augmentor.export_image(
                    image_augmented,
                    join(dir, f"{name}_{augmentation_name}{ext}"))
                to_display[augmentation_name] = image_augmented
                count += 1
    if display:
        plot_images(to_display)


def generate_balanced_dataset(tab, count, dir):
    target_per_class = min(count.values()) * (len(AUGMENTATIONS) + 1)
    missing = determine_missing(count, target_per_class)
    per_image = determine_augmentation_per_images(count, missing)

    for _cls, files in tab.items():
        log(f"Generating {missing[_cls]} augmented images for {_cls}..")

        generate_augmentation(
            join(dir, _cls), files,
            per_image[_cls], missing[_cls])


def split_dataset(dataset_path: str, final_prefix: str, split: float):
    data = Loader(dataset_path).parse()

    for _class, files in data.items():
        training_path = join(f"{final_prefix}_training", _class)
        validation_path = join(f"{final_prefix}_validation", _class)

        os.makedirs(training_path, exist_ok=True)
        os.makedirs(validation_path, exist_ok=True)

        split_index = int(len(files) * split)
        for i, file in enumerate(files):
            name = os.path.basename(file)

            if i < split_index:
                shutil.copyfile(file, join(validation_path, name))
            else:
                shutil.copyfile(file, join(training_path, name))


def dist_parsing(value):
    float_v = float(value)

    if not (0.0 < float_v < 1.0):
        raise ArgumentTypeError("Float value for distribution: 0<x<1")
    else:
        return float_v


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "path",
        help="The path to the dataset or the image to be augmented")
    parser.add_argument(
        "--split",
        type=dist_parsing,
        help="If you want to seperate result into training/validation "
        "(ex=0.1; for 10%% validation, 90%% training)")

    args = parser.parse_args()
    if not (os.path.exists(args.path)):
        log("Path do not exist!") and exit()

    is_one_file = not os.path.isdir(args.path)

    if is_one_file:
        generate_augmentation("./", [args.path], 6, 6, True)
    else:
        loader = Loader(args.path)
        generate_balanced_dataset(loader.parse(), loader.count(), args.path)

    if args.split and not is_one_file:
        split_dataset(args.path, os.path.basename(args.path), args.split)


if __name__ == "__main__":
    main()
