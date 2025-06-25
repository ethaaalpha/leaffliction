import argparse
import os
import shutil
import tempfile
from contextlib import contextmanager
from Transformation import generate_transformed_dataset
from metadata import generate_metadata
from src.preprocessing.loader import Loader
from src.utils import copy_original_images


@contextmanager
def get_label_encoding_path(dataset_dir: str):
    if not os.path.exists(dataset_dir) or not os.path.isdir(dataset_dir):
        raise ValueError("Invalid image dir path!")

    root_dir = tempfile.mkdtemp()
    try:
        dataset_tmp_dir = os.path.join(root_dir, "transformed_dataset")
        csv_path = os.path.join(root_dir, "predict.csv")
        tab = Loader(dataset_dir).parse()

        generate_transformed_dataset(tab, dataset_tmp_dir)
        copy_original_images(tab, dataset_tmp_dir)
        generate_metadata(dataset_tmp_dir, csv_path)

        yield (csv_path)
    finally:
        shutil.rmtree(root_dir)


def predict_batch(model_path, dataset_dir_path, device):
    with get_label_encoding_path(dataset_dir_path) as csv_path:
        # ici t'as le csv path comme pour le train
        # pareil que pour le predict quand t'es plus dans
        # le with tout se supprime (le csv avec les images associées)
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Path to the trained model (.pt)")
    parser.add_argument("--test_dir", required=True,
                        help="Path to the dataset test dir")
    parser.add_argument("--device", default="cpu",
                        choices=["cuda", "mps", "cpu"],
                        help="The compute device:"
                        "cuda(nvidia, amd), mps(apple)")
    args = parser.parse_args()
    predict_batch(args.model, args.test_dir, args.device)
