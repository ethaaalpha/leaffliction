from io import BytesIO
import os
import random
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser, BooleanOptionalAction
from matplotlib.figure import Figure
from pandas import DataFrame
from preprocessing.loader import Loader
from preprocessing.transformators import Transformators as Trsf
from utils import copy_original_images, log, log_dynamic
from os.path import join

random.seed(42)

IMG_TRANSFORMATIONS = [Trsf.grayscale, Trsf.pseudocolored,
                       Trsf.shape_size, Trsf.pseudolandmarks, Trsf.mask]


def get_histogram_figure(dataframe: DataFrame) -> Figure:
    fig, ax = plt.subplots()

    for color in set(dataframe['color channel']):
        subset = dataframe[dataframe['color channel'] == color]
        ax.plot(
            subset['pixel intensity'],
            subset['proportion of pixels (%)'],
            label=color, color=color
        )

    ax.legend()
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Proportion (%)')

    return fig


def save_histogram_as_image(image_path: str, result_path: str):
    transformator = Trsf(image_path)

    histgram = get_histogram_figure(transformator.color_histogram())
    histgram.savefig(result_path)


def plot_images(images: dict[str, np.ndarray | Figure]):
    _, axs = plt.subplots(2, 3)

    for ax, (name, img) in zip(axs.flatten(), images.items()):
        ax.set_title(name)

        if isinstance(img, Figure):
            buf = BytesIO()
            img.savefig(buf, format='png')
            buf.seek(0)
            ax.imshow(plt.imread(buf))
            ax.axis("off")
        else:
            ax.imshow(img)
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def generate_transformation(dir, files, only_display=False):
    images = []
    to_display = {}

    os.makedirs(dir, exist_ok=True)
    for count, img_path in enumerate(files):
        transformator = Trsf(img_path)
        name, ext = Loader.get_name_ext(img_path)

        log_dynamic(f"working on: {name.ljust(42)} ({count}/{len(files)})")

        # image classic transformations
        for t in IMG_TRANSFORMATIONS:
            image_transf = t(transformator)
            transf_name = t.__name__

            if only_display:
                to_display[transf_name] = image_transf
            else:
                images.append(join(dir, f"{name}_{transf_name}{ext.lower()}"))
                transformator.export_image(image_transf, images[-1])

        # color histogram specific
        hist_data = transformator.color_histogram()
        if only_display:
            to_display["color_histogram"] = get_histogram_figure(hist_data)
        else:
            hist_data.to_csv(join(dir, f"{name}_color_histogram.csv"))

    if only_display:
        plot_images(to_display)
    return images


def generate_transformed_dataset(tab: dict[str, list[str]], dir: str):
    for _class, files in tab.items():
        log(f"Generating {len(files) * (len(IMG_TRANSFORMATIONS) + 1)} "
            "transformed images for {_class}..")

        generate_transformation(join(dir, _class), files)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "path",
        help="The path to the dataset or the image to be transformed")
    parser.add_argument(
        "--dst",
        default="transformed_directory",
        help="If the dataset is a folder where the result will be!")
    parser.add_argument(
        "--original",
        action=BooleanOptionalAction,
        help="If present originals images will also be copied!")

    args = parser.parse_args()
    if not (os.path.exists(args.path)):
        log("Path do not exist!")
        return

    is_one_file = not os.path.isdir(args.path)

    if is_one_file:
        generate_transformation("./", [args.path], True)
    else:
        tab = Loader(args.path, True).parse()
        tab[args.dst] = tab["files"]
        tab.pop("files")

        generate_transformed_dataset(tab, "./")

        if args.original:
            copy_original_images(tab, "./")


if __name__ == "__main__":
    main()
