from torchvision import transforms
from PIL import Image
import torch
from tqdm import tqdm
import pandas as pd
import ast


def compute_mean_std(csv_path):
    df = pd.read_csv(csv_path)
    transform = transforms.ToTensor()

    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_pixels = 0

    for i, row in tqdm(df.iterrows(), total=len(df)):
        paths = ast.literal_eval(row["images"])[:6]
        for path in paths:
            img = Image.open(path).convert("RGB")
            img = transform(img)

            n_pixels += img.shape[1] * img.shape[2]
            mean += img.sum(dim=(1, 2))
            std += (img ** 2).sum(dim=(1, 2))

    mean /= n_pixels
    std = (std / n_pixels - mean ** 2).sqrt()

    return mean.tolist(), std.tolist()
