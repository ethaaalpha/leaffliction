import argparse
from torch.utils.data import DataLoader
from src.model.trainer import train_model
from src.model.MiniMobileNet import MiniMobileNet
from src.data.MultimodalDataset import MultimodalDataset
import torchvision.transforms as T


def train(b_size, n_workers, device):
    model = MiniMobileNet(csv_dim=1503, n_classes=8)
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

    train_dataset = MultimodalDataset("train.csv",
                                      transform=transform, csv_dim=1503)
    train_loader = DataLoader(train_dataset, batch_size=b_size,
                              shuffle=True,
                              num_workers=n_workers, pin_memory=True)
    val_dataset = MultimodalDataset("validation.csv",
                                    transform=transform, csv_dim=1503)
    val_loader = DataLoader(val_dataset, batch_size=b_size,
                            shuffle=False,
                            num_workers=n_workers, pin_memory=True)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=300,
        lr=1e-4,
        device=device,
        save_path="best_model.pt"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Size of each batch (memory)")
    parser.add_argument("--num_workers", default=8, type=int,
                        help="Number of workers")
    parser.add_argument("--device", default="cpu",
                        choices=["cuda", "mps", "cpu"],
                        help="The compute device:"
                        "cuda(nvidia, amd), mps(apple)")
    args = parser.parse_args()

    train(args.batch_size, args.num_workers,
          args.device)
