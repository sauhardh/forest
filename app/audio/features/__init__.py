"""
Time-Frequency Feature Representation & Acoustic Augmentation

Feature pipeline:
  1. Log-Mel spectrogram  (baseline)
  2. PCEN spectrogram     (research-grade, noise-robust)

Augmentation pipeline (training only):
  3. SpecAugment          (frequency + time masking)
  4. Acoustic Mixup       (linear spectrogram blending)
  5. Background Injection (SNR-controlled ambient noise)
"""
