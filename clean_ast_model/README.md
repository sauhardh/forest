# Clean AST Bird Classifier: End-to-End System

A modular, standalone, educational implementation of the complete bird sound identification system that achieved **70.79% Validation Top-1** and **60.98% Unseen Test Top-1** across **292 wild bird species**.

---

## Clean Architecture Map

```text
clean_ast_model/
├── config.py                     # Master audio constants & training hyperparameters
│
├── data_pipeline/                # Data collection, cleaning, normalization & clip extraction
│   ├── download_metadata.py      # Fetches recording metadata from Xeno-Canto API
│   ├── filter_and_split.py       # Filters by A/B audio quality; creates leak-free 80/10/10 split
│   ├── download_audio.py         # Concurrent multi-threaded MP3 audio downloader
│   ├── normalize_audio.py        # Resamples audio to 32 kHz mono uncompressed WAV
│   └── extract_clips.py          # Bandpass filter + RMS activity detector -> 3.0s clips + CSV
│
├── features/                     # Acoustic Signal Processing & Augmentations
│   ├── mel_pcen.py               # GPU STFT -> Mel-scale -> Adaptive PCEN gain control (<10ms)
│   ├── spec_augment.py           # GPU Frequency and Time masking augmentation
│   └── dataset.py                # BirdAudioDataset + Acoustic Mixup Collator
│
├── model/                        # Deep Learning Architecture
│   ├── ast_transformer.py        # Audio Spectrogram Transformer (AST) with gradient checkpointing
│   └── loss.py                   # Class-Balanced BCE Loss with effective-number sample weighting
│
├── execution/                    # Training, Benchmark Evaluation & Real-World Prediction
│   ├── train.py                  # 2-Stage Training Loop (Warmup -> Full Fine-Tuning)
│   ├── evaluate.py               # Test/Validation benchmark evaluation with Mean Pooling
│   └── predict.py                # Single-file audio inference (Audio -> Top-K species)
│
└── README.md                     # This complete walkthrough
```

---

## The 4 Logical Pipeline Phases

### Phase 1: Data Pipeline (`data_pipeline/`)
1. **`download_metadata.py`**: Queries the Xeno-Canto API for high-resolution acoustic data from South Asia / Nepal.
2. **`filter_and_split.py`**: Enforces strict scientific standards. Drops low-SNR recordings (`q in ['C','D','E']`), discards recordings $<3.0\text{s}$, and performs a **leak-free recording-level split** (80% Train, 10% Val, 10% Test). No audio from the same field recording can ever appear in both train and test.
3. **`download_audio.py`**: Downloads raw MP3s concurrently.
4. **`normalize_audio.py`**: Converts every file to **32 kHz mono uncompressed WAV**. 32 kHz ensures avian harmonics up to 16 kHz are preserved with zero phase distortion.
5. **`extract_clips.py`**: Slices recordings into **3.0-second analysis windows** (96,000 samples). Uses a 500 Hz – 14 kHz bandpass filter and RMS thresholding so silence or wind noise without bird song is discarded. Writes out `clips_metadata.csv`.

---

### Phase 2: Feature Engineering (`features/`)
* **`mel_pcen.py`**: Standard Log-Mel spectrograms saturate during wind gusts or mic noise. **PCEN (Per-Channel Energy Normalization)** computes an adaptive IIR low-pass filter over time to measure background sound and divides it out:
  $$\text{PCEN}(t, f) = \left( \frac{E(t, f)}{(\epsilon + M(t, f))^\alpha} + \delta \right)^r - \delta^r$$
  This acts like an automated digital noise gate, boosting faint bird vocalizations.
* **`spec_augment.py`**: Zeroes out random frequency channels and time frames directly in VRAM.
* **`dataset.py`**: Applies **Acoustic Mixup**. Since sound pressure waves add linearly in nature:
  $$\tilde{x} = \lambda x_1 + (1 - \lambda) x_2, \quad \tilde{y} = \lambda y_1 + (1 - \lambda) y_2$$
  This teaches the model to recognize multiple overlapping bird calls.

---

### Phase 3: The Model Architecture (`model/`)
* **`ast_transformer.py`**: Adapts an 86M-parameter Vision Transformer to audio. Slices the spectrogram into $16 \times 16$ acoustic patches and processes them through **12 Multi-Head Self-Attention layers**.
* **Gradient Checkpointing**: Cuts activation memory by $65\%$, allowing the heavy transformer to train on 16GB GPUs without running out of memory.
* **`loss.py`**: Uses **Class-Balanced BCE Loss** ($E_n = \frac{1 - \beta^n}{1 - \beta}$). Prevents the model from ignoring rare species that have fewer recordings.

---

### Phase 4: Training & Execution (`execution/`)
* **`train.py`**: Implements the **2-Stage Fine-Tuning Recipe**:
  * *Stage 1:* Freeze bottom 8 transformer layers; train head with `lr = 5e-4` for 15 epochs ($\sim 66\%$ accuracy).
  * *Stage 2:* Unfreeze all 12 layers; fine-tune with `lr = 3e-5` down to `1e-5` ($\mathbf{70.8\%}$ accuracy).
* **`evaluate.py` & `predict.py`**: Use **Temporal Mean Pooling** across 3-second windows, smoothing out isolated noise glitches and predicting species with calibrated confidence.

---

## Direct Python Usage

```python
import sys
sys.path.insert(0, "clean_ast_model")

# 1. Train the model:
from execution.train import run_training
run_training(clips_csv="outputs/extraction/clips_metadata.csv", save_dir="checkpoints")

# 2. Evaluate a checkpoint:
from execution.evaluate import evaluate
evaluate(checkpoint_path="checkpoints/best_model.pt", split="test")

# 3. Classify any sound file:
from execution.predict import predict_bird
predict_bird(audio_path="sample.wav", checkpoint_path="checkpoints/best_model.pt")
```
