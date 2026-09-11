import torch
import torch.nn as nn
import torchvision.models as models

from audio.model import NUM_CLASSES


class EfficientNetV2Audio(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout: float = 0.4,
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


class ASTAudio(nn.Module):
    """Audio Spectrogram Transformer backbone.

    Wraps ``MIT/ast-finetuned-audioset-10-10-0.4593`` from Hugging Face and
    adapts its input/output to match the rest of the pipeline.

    The GPU pipeline produces tensors of shape ``(batch, 1, n_mels, time_frames)``.
    AST expects ``(batch, time_frames, n_mels)``, so ``forward()`` squeezes the
    channel dim and transposes before calling the transformer.
    """

    MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout: float = 0.4,
        freeze_layers: int = 8,
    ):
        super().__init__()
        from transformers import ASTForAudioClassification

        self.ast = ASTForAudioClassification.from_pretrained(
            self.MODEL_NAME if pretrained else self.MODEL_NAME,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,  # discard 527-class AudioSet head
        )

        # ── Gradient checkpointing: recompute activations during backward ────
        # Cuts activation VRAM by ~60-70% at the cost of ~20% more compute.
        # Essential for fitting a 86 M-param ViT in 4 GB VRAM.
        self.ast.gradient_checkpointing_enable()

        # ── Freeze first N encoder layers ────────────────────────────────────
        # Use named_parameters() to match by name — robust across transformers versions.
        # Matches: any param whose name contains 'embeddings', or
        #          'encoder...layer.N...' where N < freeze_layers.
        if pretrained and freeze_layers > 0:
            import re as _re
            for param_name, p in self.ast.named_parameters():
                if "embeddings" in param_name:
                    p.requires_grad = False
                elif "encoder" in param_name:
                    m = _re.search(r"layer[s]?[.\[_](\d+)", param_name)
                    if m and int(m.group(1)) < freeze_layers:
                        p.requires_grad = False

        # Inject dropout before the linear classifier head
        if dropout > 0.0:
            orig_dense = self.ast.classifier.dense
            self.ast.classifier.dense = nn.Sequential(
                nn.Dropout(p=dropout),
                orig_dense,
            )

    # The pretrained model was trained on 1024-frame spectrograms (AudioSet ~10 s clips).
    # Mismatching this causes a position-embedding shape error at runtime.
    TARGET_TIME_FRAMES = 1024

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, n_mels, time_frames)  ← GPU mel pipeline output
        # AST wants: (batch, time_frames, n_mels)
        x = x.squeeze(1).transpose(1, 2)          # (batch, time_frames, n_mels)

        # Pad (or trim) time axis to match the pretrained position embeddings.
        T = x.shape[1]
        if T < self.TARGET_TIME_FRAMES:
            pad = self.TARGET_TIME_FRAMES - T
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))   # pad time dim
        elif T > self.TARGET_TIME_FRAMES:
            x = x[:, : self.TARGET_TIME_FRAMES, :]

        outputs = self.ast(input_values=x)
        return outputs.logits


def build_model(
    name: str = "efficientnet_v2_s",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    dropout: float = 0.4,
    freeze_layers: int = 8,
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
    elif name in ("ast", "ast_base", "audio_spectrogram_transformer"):
        return ASTAudio(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            freeze_layers=freeze_layers,
        )
    else:
        raise ValueError(
            f"Unknown backbone '{name}'. "
            f"Choose 'efficientnet_v2_s', 'efficientnet_b0', or 'ast'."
        )
