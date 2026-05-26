import torch
import torch.nn as nn


class MyModel(nn.Module):

    def __init__(self, num_classes: int, in_channels: int, **kwargs):
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError


if __name__ == '__main__':
    model = MyModel(num_classes=150, in_channels=3)
    dummy = torch.randn(2, 3, 128, 128)
    out = model(dummy)
    print("Input :", dummy.shape)
    print("Output:", out.shape)
