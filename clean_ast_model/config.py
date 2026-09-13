"""
Configuration for the Audio Spectrogram Transformer (AST) Bird Sound Classifier.
Defines audio processing constants, PCEN parameters, augmentations, and training hyperparameters.
"""

from pathlib import Path

# ── Audio Parameters ────────────────────────────────────────────────────────
SAMPLE_RATE: int = 32000          # 32 kHz captures bird harmonics up to 16 kHz Nyquist
CLIP_DURATION: float = 3.0        # Window duration in seconds
CLIP_SAMPLES: int = int(SAMPLE_RATE * CLIP_DURATION)  # 96,000 samples

# ── Spectrogram Parameters ──────────────────────────────────────────────────
N_FFT: int = 2048                 # FFT window size (~64 ms frequency resolution)
HOP_LENGTH: int = 512             # Hop size (~16 ms time resolution)
N_MELS: int = 128                 # Number of Mel frequency bins
F_MIN: float = 50.0               # Minimum frequency (Hz) - cuts electrical sub-rumble
F_MAX: float = 14000.0            # Maximum frequency (Hz) - covers highest avian calls

# ── Per-Channel Energy Normalization (PCEN) Parameters ─────────────────────
# PCEN replaces standard Log-Mel compression with an adaptive AGC filter.
# Reduces steady background noise (wind, rain, mic hiss) while preserving transient bird chirps.
PCEN_S: float = 0.025             # IIR smoothing factor for background estimation
PCEN_ALPHA: float = 0.98          # Exponent for automatic gain control divisor
PCEN_DELTA: float = 2.0           # Bias offset before root compression
PCEN_R: float = 0.5               # Root compression exponent (square root compression)
PCEN_EPS: float = 1e-6            # Numerical stability epsilon

# ── Augmentation Parameters ────────────────────────────────────────────────
FREQ_MASK_PARAM: int = 36         # Max frequency channels to zero-out in SpecAugment
TIME_MASK_PARAM: int = 64         # Max time frames to zero-out in SpecAugment
NUM_FREQ_MASKS: int = 2           # Number of frequency masking blocks
NUM_TIME_MASKS: int = 2           # Number of time masking blocks

MIXUP_ALPHA: float = 0.4          # Beta distribution parameter for linear audio mixup
MIXUP_PROB: float = 0.5           # Probability of applying mixup to a batch

# ── Model Architecture ──────────────────────────────────────────────────────
PRETRAINED_AST_NAME: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_TARGET_FRAMES: int = 1024     # AST pretrained positional embedding frame length
NUM_CLASSES: int = 292            # Total bird species classes

# ── Stage 1 Training Hyperparameters (Transfer Learning / Warmup) ──────────
STAGE1_LR: float = 5e-4
STAGE1_FREEZE_LAYERS: int = 8     # Freeze first 8 of 12 transformer encoder layers
STAGE1_EPOCHS: int = 15

# ── Stage 2 Training Hyperparameters (Full Fine-Tuning) ────────────────────
STAGE2_LR: float = 3e-5           # 10x smaller learning rate to prevent catastrophic forgetting
STAGE2_FREEZE_LAYERS: int = 0     # Unfreeze all 12 layers for end-to-end feature adaptation
STAGE2_EPOCHS: int = 10
STAGE2_DROPOUT: float = 0.5
STAGE2_WEIGHT_DECAY: float = 1e-2

# ── Hardware & Dataloader ──────────────────────────────────────────────────
BATCH_SIZE: int = 16
NUM_WORKERS: int = 4
PIN_MEMORY: bool = True
