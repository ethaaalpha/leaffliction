import argparse
import os
import shutil
import tempfile
import torch
from tqdm import tqdm
from contextlib import contextmanager
from Transformation import generate_transformed_dataset
from metadata import generate_metadata
from src.preprocessing.loader import Loader
from src.utils import copy_original_images
from src.data.MultimodalDataset import MultimodalDataset
from src.model.MiniMobileNet import MiniMobileNet
from torch.utils.data import DataLoader
from torchvision import transforms as T


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


def predict_batch(model_path, dataset_dir_path, device,
                  labels, batch_size, num_workers):
    with get_label_encoding_path(dataset_dir_path) as csv_path:
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.37272489070892334,
                              0.3830896019935608,
                              0.3935730457305908],
                        std=[0.31965553760528564,
                             0.3071448802947998,
                             0.3268057703971863])
        ])
        dataset = MultimodalDataset(
            csv_path=csv_path,
            transform=transform,
            csv_dim=1503,
            labels_json_path=labels
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        model = MiniMobileNet(csv_dim=1503, n_classes=8)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, hist, labels in tqdm(dataloader, desc="testing"):
                imgs = [img.to(device) for img in imgs]
                hist = hist.to(device)
                labels = labels.to(device)

                outputs = model(imgs, hist)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total if total > 0 else 0
        print(f"Accuracy: {accuracy*100:.4f}%")
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
    parser.add_argument("--labels", required=True,
                        help="Path to the labels json file")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for prediction")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of workers for DataLoader")
    args = parser.parse_args()
    predict_batch(args.model, args.test_dir, args.device,
                  args.labels, args.batch_size, args.num_workers)
