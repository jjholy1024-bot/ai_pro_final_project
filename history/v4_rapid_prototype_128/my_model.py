import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel-wise attention"""
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
    """Deep Convolutional Block (2 Conv layers) with optional SE"""
    def __init__(self, in_channels, out_channels, use_se=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.se(x)
        return self.pool(x)

class FeatureExtractor(nn.Module):
    """5-Block Deep Feature Extractor"""
    def __init__(self, in_channels=3, use_se=False):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, use_se),   # 192 -> 96
            ConvBlock(32, 64, use_se),            # 96 -> 48
            ConvBlock(64, 128, use_se),           # 48 -> 24
            ConvBlock(128, 256, use_se),          # 24 -> 12
            ConvBlock(256, 512, use_se),          # 12 -> 6
            nn.AdaptiveAvgPool2d(1)               # 6 -> 1 (GAP)
        )

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

class MyModel(nn.Module):
    """
    Golden Standard K-Food Model
    - 5-Block Depth for complex feature extraction
    - GAP (Global Average Pooling) for parameter efficiency & generalization
    - Modular: toggle SE and Ensemble via YAML
    """
    def __init__(self, num_classes: int, in_channels: int = 3, dropout: float = 0.5, **kwargs):
        super(MyModel, self).__init__()
        
        self.use_se = kwargs.get('use_se', False)
        self.ensemble_count = kwargs.get('ensemble_count', 1)
        
        # 멀티 컬럼 구조
        self.columns = nn.ModuleList([
            FeatureExtractor(in_channels, self.use_se) for _ in range(self.ensemble_count)
        ])
        
        # GAP 덕분에 분류기가 매우 단순해짐 (과적합 방지)
        combined_dim = 512 * self.ensemble_count
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # 각 컬럼에서 특징 추출
        feats = [col(x) for col in self.columns]
        # 특징 결합
        feat = torch.cat(feats, dim=1)
        # 최종 분류
        return self.classifier(feat)

if __name__ == '__main__':
    # 5층 + GAP 모델 테스트
    model = MyModel(num_classes=150, use_se=True, ensemble_count=1)
    dummy = torch.randn(2, 3, 192, 192)
    out = model(dummy)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: 5-Blocks + GAP | Params: {param_count:,}")
    print(f"Input: {dummy.shape} -> Output: {out.shape}")
