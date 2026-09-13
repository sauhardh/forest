# Audio Spectrogram Transformer (AST) Bird Sound Classifier

A clean, standalone, educational implementation of the **Audio Spectrogram Transformer (AST)** pipeline that achieved **70.79% Validation Top-1** and **60.98% Unseen Test Top-1** across **292 wild bird species**.

---

## Architecture Overview

```
Raw Waveform (32 kHz, 3.0s) [96,000 samples]
     │
     ▼
GPU Mel-Spectrogram (128 mels, 188 frames)
     │
     ▼
Per-Channel Energy Normalization (PCEN)
     │
     ▼
SpecAugment (Frequency & Time Masking)
     │
     ▼
Transpose & Pad to 1024 frames: (B, 1024, 128)
     │
     ▼
AST Backbone (16x16 Patch Embedding + 12 Transformer Layers)
     │
     ▼
Linear Classifier Head (768 -> 292 Species)
     │
     ▼
Sigmoid -> Temporal Mean Pooling -> Final Predictions
```

---

## File Structure

| File | Purpose | Key Concept |
| :--- | :--- | :--- |
| **`config.py`** | Central configuration | All audio, model, and training hyperparameters in one place. No CLI flags. |
| **`features.py`** | GPU Feature Extraction | `TorchMelPCEN` and `TorchSpecAugment` computed in VRAM in $<10\text{ ms}$. |
| **`dataset.py`** | Audio loading & Mixup | Loads 3.0s clips, generates one-hot vectors, and blends audio waveforms. |
| **`model.py`** | AST Transformer | Wraps MIT AudioSet ViT, adapts patch dimensions, and enables gradient checkpointing. |
| **`loss.py`** | Class-Balanced BCE | Weights rare species higher using effective number weighting ($E_n$). |
| **`train.py`** | 2-Stage Training Loop | Implements the exact 2-stage training schedule that broke the 70% barrier. |
| **`predict.py`** | Single-file inference | Slices any audio recording, runs inference, and prints top species. |
| **`evaluate.py`** | Split evaluation | Calculates Clip Top-1/5 and Recording Top-1/5 using temporal mean pooling. |

---

## The 5 Core Concepts to Learn

### 1. Why PCEN instead of Standard Log-Mel?
Standard Mel spectrograms use $\log(E + \epsilon)$ compression. In forest recordings, loud wind or microphone hiss saturates the log spectrogram, hiding quiet bird chirps.
**PCEN (Per-Channel Energy Normalization)** computes an adaptive low-pass filter $M(t)$ that tracks ambient background noise and divides it out:
$$\text{PCEN}(t, f) = \left( \frac{E(t, f)}{(\epsilon + M(t, f))^\alpha} + \delta \right)^r - \delta^r$$
This acts like an automated noise gate, making faint, distant bird vocalizations stand out clearly.

### 2. Why Acoustic Mixup?
Sound pressure waves are physically additive. Blending two audio files:
$$\tilde{x} = \lambda x_1 + (1 - \lambda) x_2, \quad \tilde{y} = \lambda y_1 + (1 - \lambda) y_2$$
physically mimics two birds calling simultaneously in the same forest. This regularizes the model and teaches it to identify overlapping bird songs.

### 3. Why AST (Vision Transformer for Audio)?
Traditional CNNs (like ResNet or EfficientNet) use small local convolution kernels (e.g. $3 \times 3$). Bird calls, however, have long-range harmonic relationships: a high-frequency chirp at 8 kHz is directly coupled to its fundamental tone at 2 kHz.
**AST treats the spectrogram as an image**, breaks it into $16 \times 16$ patches, and uses **Self-Attention** to compute correlations across all frequency bands and time steps simultaneously.

### 4. Gradient Checkpointing
A 12-layer Vision Transformer with 86 million parameters produces massive activation maps that can easily cause Out-Of-Memory (OOM) errors.
`self.ast.gradient_checkpointing_enable()` discards intermediate activations during the forward pass and recomputes them on-the-fly during the backward pass. This reduces activation VRAM by **65%**, allowing batch size 16 to train smoothly on standard 16GB GPUs.

### 5. The 2-Stage Fine-Tuning Recipe
* **Stage 1 (Transfer Warmup):** 
  * Freeze the bottom 8 layers of the transformer.
  * Train only the top 4 layers and the classification head at a relatively high learning rate (`lr = 5e-4`).
  * *Result:* Fast alignment to bird sounds without breaking general acoustic features.
* **Stage 2 (Full Fine-Tuning):**
  * Unfreeze **all 12 layers**.
  * Drop the learning rate by $10\times$ (`lr = 3e-5` down to `1e-5`) with stronger weight decay (`1e-2`).
  * *Result:* Allows low-level attention heads to adapt to avian frequencies, pushing accuracy from $66\%$ to **$70.8\%$**.

---

## How to Run

### To Train:
```python
from train import run_training

# Stage 1: Warmup
run_training(clips_csv="path/to/clips_metadata.csv", save_dir="checkpoints")

# Stage 2: Full Fine-Tuning (resuming from Stage 1 checkpoint)
run_training(
    clips_csv="path/to/clips_metadata.csv",
    save_dir="checkpoints",
    resume_checkpoint="checkpoints/best_model.pt"
)
```

### To Evaluate a Test Set:
```python
from evaluate import evaluate

evaluate(checkpoint_path="checkpoints/best_model.pt", split="test", pool_method="mean")
```

### To Classify Any Audio Recording:
```python
from predict import predict_bird

predict_bird(
    audio_path="my_recording.wav",
    checkpoint_path="checkpoints/best_model.pt",
    top_k=5
)
```
