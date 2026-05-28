import torch
import torch.nn as nn


class MyModel(nn.Module):

    def __init__(self, num_classes: int, in_channels: int, **kwargs):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = MyModel(num_classes=150, in_channels=3)
    dummy = torch.randn(2, 3, 128, 128)
    out = model(dummy)
    print("Input :", dummy.shape)
    print("Output:", out.shape)
