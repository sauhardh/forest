"""
Phase 5: Deep Learning Architecture & Spatial-Ecological Integration
Components:
  dataset.py  – BirdDataset (torch Dataset) + DataLoader factory
  backbone.py – EfficientNet-V2-S (CNN) and AST (Transformer) wrappers
  loss.py     – Class-Balanced Loss and Focal Loss for long-tail imbalance
  fusion.py   – Geo-Ecological MLP + product-of-experts fusion
  train.py    – Training loop, validation, checkpointing
"""

NUM_CLASSES = 304
ECO_DIM = 4  # [lat, lon, elevation, ndvi]
