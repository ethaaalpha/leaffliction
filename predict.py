import argparse
import os
import shutil
import tempfile
import torch
import sys
from contextlib import contextmanager
from torchvision import transforms as T
from src.data.MultimodalDataset import MultimodalDataset
import json
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append("./src")
from metadata import generate_metadata
from Transformation import generate_transformation, save_histogram_as_image
from preprocessing.loader import Loader
from model.MiniMobileNet import MiniMobileNet


@contextmanager
def get_tmp_images(image_path: str):
    if not os.path.exists(image_path) or not os.path.isfile(image_path):
        raise ValueError("Invalid image path!")

    root_dir = tempfile.mkdtemp()
    try:
        _, ext = Loader.get_name_ext(image_path)
        transf_dir = os.path.join(root_dir, "data")
        tmp_image = os.path.join(transf_dir, f"image{ext}")
        histogram_image = os.path.join(root_dir, "color_histogram.jpg")
        csv_file = os.path.join(root_dir, "predict.csv")

        os.makedirs(transf_dir)
        shutil.copyfile(image_path, tmp_image)

        images_transformed = generate_transformation(transf_dir, [tmp_image])
        save_histogram_as_image(tmp_image, histogram_image)
        generate_metadata(transf_dir, csv_file, True)

        yield (csv_file, [*images_transformed, histogram_image])
    finally:
        shutil.rmtree(root_dir)


def predict(model_path, image_path, label_encoding,
            device='cuda' if torch.cuda.is_available() else 'cpu'):
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

    model = MiniMobileNet(csv_dim=1503, n_classes=8)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with open(label_encoding, "r") as f:
        label_maps = json.load(f)
        index_to_label = {int(k): v for k,
                          v in label_maps["index_to_label"].items()}

    with get_tmp_images(image_path) as (csv_path, images_path):
        predict_data = MultimodalDataset(csv_path, transform=transform,
                                         csv_dim=1503,
                                         labels_json_path=label_encoding)
        imgs_tensor, hist_tensor, _ = predict_data.__getitem__(0)
        imgs_tensor = [img.unsqueeze(0).to(device) for img in imgs_tensor]
        hist_tensor = hist_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(imgs_tensor, hist_tensor)
            pred = output.argmax(dim=1).item()
            predicted_class = index_to_label[pred]

        fig, axs = plt.subplots(2, 4, figsize=(16, 8))
        images_path.append(image_path)
        for idx, image_path in enumerate(images_path):
            if idx >= 8:
                break
            img = Image.open(image_path).convert("RGB")
            ax = axs[idx // 4][idx % 4]
            ax.imshow(img)
            ax.axis("off")
            if idx < 6:
                ax.set_title(f"Transfo {idx+1}")
            else:
                ax.set_title("Original Image")
        axs[-1][-1].axis('off')
        fig.suptitle(f"Predicted class: {predicted_class}", fontsize=20)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Path to the trained model (.pt)")
    parser.add_argument("--image", required=True,
                        help="Path to the input image")
    parser.add_argument("--label_encoding", required=True,
                        help="Path to the label.json file")
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predict(args.model, args.image, args.label_encoding, device=device)
