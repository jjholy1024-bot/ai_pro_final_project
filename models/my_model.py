import torch
import torch.nn as nn

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
    """
    기본 합성곱 블록
    - Conv -> BN -> ReLU (2회 반복)
    - SEBlock: 채널 어텐션 적용
    - Dropout2d: 공간적 과적합 방지
    """
    def __init__(self, in_channels, out_channels, use_se=True, dropout=0.1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.drop(self.se(self.conv(x))))

class HeteroFeatureExtractor(nn.Module):
    """이종 특징 추출기: 블록 수 및 채널 구성을 유연하게 조정"""
    def __init__(self, in_channels=3, blocks=5, use_se=True):
        super(HeteroFeatureExtractor, self).__init__()
        layers = []
        # 채널 수를 점진적으로 확장 (6블록 이상 대응 가능하도록 설계)
        channels = [32, 64, 128, 256, 512, 512, 1024] 
        curr_in = in_channels
        
        for i in range(blocks):
            # 깊은 층으로 갈수록 드롭아웃 비중을 소폭 상향하여 일반화 성능 확보
            drop_rate = 0.1 if i < 3 else 0.2
            layers.append(ConvBlock(curr_in, channels[i], use_se, dropout=drop_rate))
            curr_in = channels[i]
            
        # Global Average Pooling을 통해 파라미터 수 절감 및 특징 압축
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*layers)
        self.out_dim = channels[blocks-1]

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

class MyModel(nn.Module):
    """
    70% 달성을 위한 최종 하이브리드 모델
    - Heterogeneous Ensemble: 서로 다른 수용 영역(Receptive Field)을 가진 컬럼 조합
    - Spatial Regularization: Dropout2d를 통한 강력한 과적합 억제
    - Deep Classifier: 특징 통합을 위한 다층 퍼셉트론 구조
    """
    def __init__(self, num_classes: int, in_channels: int = 3, dropout: float = 0.5, 
                 ensemble_blocks: list = [5, 6], use_se: bool = True, **kwargs):
        super(MyModel, self).__init__()
        
        # 설정된 블록 수에 따라 독립적인 특징 추출 컬럼 생성
        self.columns = nn.ModuleList([
            HeteroFeatureExtractor(in_channels, blocks=b, use_se=use_se) 
            for b in ensemble_blocks
        ])
        
        combined_dim = sum([col.out_dim for col in self.columns])
        
        # 앙상블된 특징을 통합하여 최종 분류를 수행하는 헤드
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(dropout),
            
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # 모든 컬럼에서 특징 추출 후 채널 방향으로 결합
        features = [col(x) for col in self.columns]
        feat = torch.cat(features, dim=1)
        return self.classifier(feat)

if __name__ == '__main__':
    model = MyModel(num_classes=150)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Hetero Ensemble Model Loaded. Output: {out.shape}")
