import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class MiniConv(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            DepthwiseSeparableConv(16, 32, stride=1),
            DepthwiseSeparableConv(32, out_dim, stride=2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

class MiniMobileNet(nn.Module):
    def __init__(self, csv_dim=1503, n_classes=8, img_feat_dim=64):
        super().__init__()
        self.encoder_original      = MiniConv(img_feat_dim)
        self.encoder_grayscale     = MiniConv(img_feat_dim)
        self.encoder_pseudocolor   = MiniConv(img_feat_dim)
        self.encoder_shapesize     = MiniConv(img_feat_dim)
        self.encoder_landmarks     = MiniConv(img_feat_dim)
        self.encoder_mask          = MiniConv(img_feat_dim)

        self.csv_proj = nn.Sequential(
            nn.Linear(csv_dim, img_feat_dim * 2),
            nn.ReLU()
        )

        total_feat = 6 * img_feat_dim + img_feat_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(total_feat, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes)
        )

    def forward(self, imgs, csv):
        f1 = self.encoder_original(imgs[0])
        f2 = self.encoder_grayscale(imgs[1])
        f3 = self.encoder_pseudocolor(imgs[2])
        f4 = self.encoder_shapesize(imgs[3])
        f5 = self.encoder_landmarks(imgs[4])
        f6 = self.encoder_mask(imgs[5])

        csv_feat = self.csv_proj(csv)

        fused = torch.cat([f1, f2, f3, f4, f5, f6, csv_feat], dim=1)
        return self.classifier(fused)
