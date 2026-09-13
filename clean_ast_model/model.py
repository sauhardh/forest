"""
Audio Spectrogram Transformer (AST) Architecture.
Adapts MIT's AudioSet-pretrained Vision Transformer for 3.0s PCEN bird sound spectrograms.
"""

import re
import torch
import torch.nn as nn
from transformers import ASTForAudioClassification

from config import (
    PRETRAINED_AST_NAME,
    AST_TARGET_FRAMES,
    NUM_CLASSES,
    STAGE1_FREEZE_LAYERS,
)


class ASTBirdClassifier(nn.Module):
    """
    Audio Spectrogram Transformer (AST) for fine-grained bird call identification.
    
    Transforms (Batch, 1, 128, Frames) PCEN spectrograms into a sequence of
    16x16 acoustic patches, processed through 12 multi-head self-attention layers.
    """
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.5,
        freeze_layers: int = STAGE1_FREEZE_LAYERS,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.target_time_frames = AST_TARGET_FRAMES

        # 1. Load pretrained AudioSet backbone (discards original 527-class head)
        self.ast = ASTForAudioClassification.from_pretrained(
            PRETRAINED_AST_NAME,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

        # 2. Gradient Checkpointing: Recomputes activations during backward pass.
        # Cuts VRAM consumption by ~65%, allowing the 86M ViT to train with large batches.
        self.ast.gradient_checkpointing_enable()

        # 3. Inject Dropout before dense classification head
        if dropout > 0.0:
            orig_dense = self.ast.classifier.dense
            self.ast.classifier.dense = nn.Sequential(
                nn.Dropout(p=dropout),
                orig_dense,
            )

        # 4. Freeze bottom layers for initial stage warmup
        if freeze_layers > 0:
            self.freeze_bottom_layers(freeze_layers)

    def freeze_bottom_layers(self, n_layers: int):
        """Freezes patch embeddings and the first N of 12 transformer encoder layers."""
        for name, param in self.ast.named_parameters():
            if "embeddings" in name:
                param.requires_grad = False
            elif "encoder" in name:
                match = re.search(r"layer[s]?[.\[_](\d+)", name)
                if match and int(match.group(1)) < n_layers:
                    param.requires_grad = False

    def unfreeze_all_layers(self):
        """Unfreezes the entire transformer for Stage 2 full fine-tuning."""
        for param in self.ast.parameters():
            param.requires_grad = True

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec: (Batch, 1, 128, Time_Frames) from TorchMelPCEN
        Returns:
            logits: (Batch, Num_Classes) unnormalized classification logits
        """
        # AST expects input shape: (Batch, Time_Frames, N_MELS)
        x = spec.squeeze(1).transpose(1, 2)

        # Standardize time length to match AST's 1024-frame positional embeddings
        t_frames = x.shape[1]
        if t_frames < self.target_time_frames:
            pad = self.target_time_frames - t_frames
            x = nn.functional.pad(x, (0, 0, 0, pad))
        elif t_frames > self.target_time_frames:
            x = x[:, : self.target_time_frames, :]

        # Forward pass through Transformer encoder and classification head
        outputs = self.ast(input_values=x)
        return outputs.logits
