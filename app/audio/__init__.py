"""
1. metadata/get_metadata
    -> Metadata + link to download audio

2. metadata/metadata_filter
    -> filter based on quality,
    -> Remove low numbered data

3. split_dataset
    -> Train (80%), Test (10%), Val (10%)

4. download_audio
    -> Download those audio according with Train, Test, Val

5. process/normalize_audio
    -> Normalize into standard 32KHz mono uncompressed WAV format

6. analyze_audio
    -> For Testing

7. process/extract_clips
    ->
"""

import os
import sys
from pathlib import Path

# Ensure 'app' directory is in sys.path so submodules can import 'audio' regardless of cwd
_app_dir = str(Path(__file__).resolve().parent.parent)
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)


def get_base_output_dir() -> Path:
    """
    Resolves the base output directory dynamically:
    1. FOREST_OUTPUT_DIR environment variable (useful for Colab / Google Drive).
    2. outputs/ directory in current working directory.
    3. outputs/ directory in the app/ parent directory (default local setup).
    """
    if "FOREST_OUTPUT_DIR" in os.environ:
        p = Path(os.environ["FOREST_OUTPUT_DIR"]).resolve()
        p.mkdir(parents=True, exist_ok=True)
        # If the user unpacked a zip that created a nested outputs/ directory
        if (p / "outputs" / "extraction").exists() and not (p / "extraction").exists():
            return (p / "outputs").resolve()
        if (p / "outputs" / "splits").exists() and not (p / "splits").exists():
            return (p / "outputs").resolve()
        return p


    cwd_outputs = Path.cwd() / "outputs"
    if cwd_outputs.exists():
        return cwd_outputs.resolve()

    app_outputs = Path(__file__).resolve().parent.parent / "outputs"
    if app_outputs.exists():
        return app_outputs.resolve()

    # Fallback to local app/outputs
    return app_outputs.resolve()


BASE_OUTPUT_DIR = get_base_output_dir()

SAMPLE_RATE = 32_000

# Download audio
SPLIT_DIR = BASE_OUTPUT_DIR / "splits"
AUDIO_DIR = BASE_OUTPUT_DIR / "audio"

# Split_dataset
INPUT_PATH = BASE_OUTPUT_DIR / "nepal_bird_filtered.csv"
OUTPUT_DIR = BASE_OUTPUT_DIR / "splits"

# Normalize_audio
RAW_DIR = BASE_OUTPUT_DIR / "audio"
PROCESSED_DIR = BASE_OUTPUT_DIR / "processed" / "audio"

# metadata_filter
METADATA_PATH = BASE_OUTPUT_DIR / "nepal_bird_recordings.csv"
ANNOTATIONS_PATH = BASE_OUTPUT_DIR / "nepal_bird_annotations.csv"
OUTPUT_PATH = BASE_OUTPUT_DIR / "nepal_bird_filtered.csv"

# Analyze audio
ANALYZE_PATH = BASE_OUTPUT_DIR / "processed" / "csv" / "audio_analysis.csv"

# Extract clips
CLIPS_DIR = BASE_OUTPUT_DIR / "extraction" / "clips"
CLIPS_METADATA_PATH = BASE_OUTPUT_DIR / "extraction" / "clips_metadata.csv"

# Checkpoints
CHECKPOINTS_DIR = BASE_OUTPUT_DIR / "checkpoints"

