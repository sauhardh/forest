# 🌲 Forest: Acoustic Monitoring & Bird Species Classification in Nepal

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sauhardh/forest/blob/main/forest_colab.ipynb)

Deep learning-based bioacoustic monitoring and species classification for over 300 bird species across Nepal and the Himalayas.

---

## ⚡ Quickstart on Google Colab (1-Click Setup)

You can run the entire pipeline and train models directly on Google Colab with GPU acceleration in a few clicks:

1. Click the **[Open In Colab](https://colab.research.google.com/github/sauhardh/forest/blob/main/forest_colab.ipynb)** badge.
2. In Colab, make sure you have GPU enabled: **Runtime** → **Change runtime type** → **T4 GPU**.
3. Run through the cells step-by-step:
   - **Step 1**: Check GPU acceleration.
   - **Step 2**: Clone repository & install dependencies.
   - **Step 3**: (Recommended) Mount Google Drive to persist checkpoints and datasets across sessions.
   - **Step 4**: Prepare data (use existing clips or run the automated pipeline).
   - **Step 5**: Train the model with live metrics and auto-checkpointing.
   - **Step 6**: Export and download the best trained checkpoint.

---

## 💻 Local Setup

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/sauhardh/forest.git
cd forest

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements and package
pip install -r requirements.txt
pip install -e .
```

### 2. Environment Variables (Optional)

You can specify where data and outputs are stored by setting `FOREST_OUTPUT_DIR`. By default, it will detect `outputs/` in your current working directory or `app/outputs`:

```bash
# Optional: point to custom storage or external drive
export FOREST_OUTPUT_DIR=/path/to/my/outputs
```

---

## 🔬 Pipeline Workflow

1. **Phase 1: Metadata Retrieval** (`app/audio/metadata/get_metadata.py`)  
   Retrieves bird sound metadata and recordings for Nepal from Xeno-Canto.
2. **Phase 2: Metadata Filtering** (`app/audio/metadata/metadata_filter.py`)  
   Filters recordings by quality (A, B, C) and minimum duration.
3. **Phase 3: Stratified Split** (`app/audio/split_dataset.py`)  
   Splits dataset into 80% Train, 10% Validation, and 10% Test without data leakage.
4. **Phase 4: Audio Download** (`app/audio/download_audio.py`)  
   Downloads MP3 recordings from Xeno-Canto for each split.
5. **Phase 5: Audio Normalization** (`app/audio/process/normalize_audio.py`)  
   Converts audio to standardized 32 kHz mono uncompressed WAV format.
6. **Phase 6: Bird Activity Detection & Clip Slicing** (`app/audio/process/extract_clips.py`)  
   Detects bird sound using bandpass filtering and RMS thresholding, slicing into 3-second clips.
7. **Phase 7: Deep Learning Training** (`app/audio/model/train.py`)  
   Trains an `EfficientNet-V2-S` backbone with PCEN / log-Mel spectrogram transforms, SpecAugment, Acoustic Mixup, and Class-Balanced loss.

```bash
# Train the model with custom parameters
python -m audio.model.train --epochs 25 --batch-size 16 --lr 5e-4 --backbone efficientnet_v2_s
```

---

## 📁 Repository Structure

```
forest/
├── README.md                  # Project documentation & Colab launcher
├── forest_colab.ipynb         # 1-click Google Colab notebook
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup configuration
└── app/
    ├── pyproject.toml         # Package definition & dependencies
    ├── audio/
    │   ├── __init__.py        # Dynamic path resolution & constants
    │   ├── download_audio.py  # Audio downloader
    │   ├── split_dataset.py   # Stratified splitting
    │   ├── metadata/          # Metadata query & filtering
    │   ├── process/           # Resampling & clip extraction
    │   ├── features/          # Mel transform, PCEN, SpecAugment, Mixup
    │   └── model/             # Backbones, loss functions, training loop
    └── src/
        └── satellite/         # Earth Engine satellite NDVI integration
```

---

## 📜 License
MIT License.
