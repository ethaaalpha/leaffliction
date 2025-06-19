from argparse import ArgumentParser
from preprocessing.augmentors import Augmentators as Augs
from preprocessing.transformators import Transformators as Trsf
from preprocessing.loader import Loader
from os import path
import pandas as pd

AUGMENTATIONS = [Augs.rotate, Augs.contrast, Augs.brightness, Augs.projective, Augs.blur, Augs.scale]
TRANSFORMATIONS = [Trsf.grayscale, Trsf.pseudocolored, Trsf.shape_size, Trsf.pseudolandmarks, Trsf.mask, Trsf.color_histogram]

def find_transformations(image_path: str):
    name, ext = Loader.get_name_ext(image_path)
    dirname = path.dirname(image_path)
    result = [image_path]

    for tr in TRANSFORMATIONS:
        if tr == Trsf.color_histogram:
            ext = ".csv" 
        filepath = f"{dirname}/{name}_{tr.__name__}{ext}"

        if path.exists(filepath):
            result.append(filepath)
    return result

def runner(dataset: str, result: str):
    rows = []
    load = Loader(dataset)
    for _class, files in load.parse().items():
        for file in files:
            img_name = path.basename(file)
            split = img_name.split("_")

            if len(split) == 1: # original img
                rows.append({
                    "class": _class,
                    "original": img_name,
                    "images": find_transformations(file)
                })
            if len(split) == 2: # maybe an original augmented image
                transformations = find_transformations(file)

                if len(transformations) == 7:
                    rows.append({
                        "class": _class,
                        "original": img_name,
                        "images": transformations
                    })

    df = pd.DataFrame(rows)
    df.to_csv(result, index=False)

def main():
    parser = ArgumentParser()
    parser.add_argument("path", help="The path to the dataset or the image to be transformed")
    parser.add_argument("csv", help="The name of the csv file result")

    args = parser.parse_args()
    runner(args.path, args.csv)

if __name__ == "__main__":
    main()