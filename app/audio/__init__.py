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

from pathlib import Path


SAMPLE_RATE = 32_000
# Download audio
SPLIT_DIR = Path("outputs/splits")
AUDIO_DIR = Path("outputs/audio")

# Split_dataset
INPUT_PATH = Path("outputs/nepal_bird_filtered.csv")
OUTPUT_DIR = Path("outputs/splits")

# Normalize_audio
RAW_DIR = Path("outputs/audio")
PROCESSED_DIR = Path("outputs/processed/audio")

# metadata_filter
METADATA_PATH = Path("outputs/nepal_bird_recordings.csv")
ANNOTATIONS_PATH = Path("outputs/nepal_bird_annotations.csv")
OUTPUT_PATH = Path("outputs/nepal_bird_filtered.csv")

# Analyze audio
ANALYZE_PATH = Path("outputs/processed/csv/audio_analysis.csv")

# Extract clips
CLIPS_DIR = Path("outputs/extraction/clips")
CLIPS_METADATA_PATH = Path("outputs/extraction/clips_metadata.csv")
