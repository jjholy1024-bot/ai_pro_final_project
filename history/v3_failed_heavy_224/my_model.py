import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEBlock(out_channels),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.conv(x)

class EnsembleColumn(nn.Module):
    def __init__(self, in_channels=3):
        super(EnsembleColumn, self).__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

class MyModel(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3, dropout: float = 0.5):
        super(MyModel, self).__init__()
        self.column1 = EnsembleColumn(in_channels)
        self.column2 = EnsembleColumn(in_channels)
        self.column3 = EnsembleColumn(in_channels)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        f1 = self.column1(x)
        f2 = self.column2(x)
        f3 = self.column3(x)
        feat = torch.cat([f1, f2, f3], dim=1)
        return self.classifier(feat)
