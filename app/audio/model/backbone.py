import torch
import torch.nn as nn
import torchvision.models as models

from audio.model import NUM_CLASSES


class EfficientNetV2Audio(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes

        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        base = models.efficientnet_v2_s(weights=weights)

        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        in_features = base.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(in_features, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        feat_map = self.features(x)
        pooled = self.pool(feat_map)
        return torch.flatten(pooled, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.extract_features(x)
        return self.classifier(emb)


class EfficientNetB0Audio(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base = models.efficientnet_b0(weights=weights)

        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        in_features = base.classifier[1].in_features

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        feat = self.pool(self.features(x))

        return self.classifier(torch.flatten(feat, 1))


def build_model(
    name: str = "efficientnet_v2_s",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    name = name.lower()
    if name in ("efficientnet_v2_s", "v2_s", "effnet_v2"):
        return EfficientNetV2Audio(
            num_classes=num_classes, pretrained=pretrained, dropout=dropout
        )
    elif name in ("efficientnet_b0", "b0", "effnet_b0"):
        return EfficientNetB0Audio(
            num_classes=num_classes, pretrained=pretrained, dropout=dropout
        )
    else:
        raise ValueError(
            f"Unknown backbone '{name}'. Choose 'efficientnet_v2_s' or 'efficientnet_b0'."
        )
