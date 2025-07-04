import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import ast
import json


class MultimodalDataset(Dataset):
    def __init__(self, csv_path, transform=None, csv_dim=1503,
                 labels_json_path="labels.json"):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.csv_dim = csv_dim

        with open(labels_json_path, "r") as f:
            label_map = json.load(f)
            self.label_to_index = label_map["label_to_index"]
            self.index_to_label = {int(k): v for k,
                                   v in label_map["index_to_label"].items()}
        if "class" in self.df.columns and self.df["class"].notnull().all():
            self.df["label"] = self.df["class"].map(self.label_to_index)
        else:
            self.df["label"] = -1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row['label'])

        paths = ast.literal_eval(row['images'])
        img_paths, hist_path = paths[:6], paths[6]

        imgs = [Image.open(p).convert("RGB") for p in img_paths]
        if self.transform:
            imgs = [self.transform(img) for img in imgs]

        hist_df = pd.read_csv(hist_path, header=None, skiprows=1)

        hist_r = pd.to_numeric(hist_df[2][hist_df[4] == 'red'],
                               errors='coerce').fillna(0).values
        hist_g = pd.to_numeric(hist_df[2][hist_df[4] == 'green'],
                               errors='coerce').fillna(0).values
        hist_b = pd.to_numeric(hist_df[2][hist_df[4] == 'blue'],
                               errors='coerce').fillna(0).values

        max_len = self.csv_dim // 3

        def pad_or_truncate(arr):
            if len(arr) < max_len:
                return np.pad(arr, (0, max_len - len(arr)))
            else:
                return arr[:max_len]

        hist_r = pad_or_truncate(hist_r)
        hist_g = pad_or_truncate(hist_g)
        hist_b = pad_or_truncate(hist_b)

        hist = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32)
        hist = torch.tensor(hist, dtype=torch.float32)

        return imgs, hist, label
