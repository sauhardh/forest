"""
Step 4: Audio Normalization.
Converts arbitrary MP3/OGG recordings into standardized 32 kHz mono uncompressed WAV files.
"""

from pathlib import Path
import librosa
import soundfile as sf

TARGET_SAMPLE_RATE = 32000


def normalize_recording(input_path: Path, output_path: Path, sample_rate: int = TARGET_SAMPLE_RATE) -> bool:
    """Loads arbitrary audio, converts to single-channel mono at 32 kHz, and saves as WAV."""
    if output_path.exists():
        return True

    try:
        waveform, _ = librosa.load(str(input_path), sr=sample_rate, mono=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), waveform, sample_rate)
        return True
    except Exception as e:
        print(f"Error normalizing {input_path.name}: {e}")
        return False


def normalize_all_audio(raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
    """Processes all splits from raw MP3 into 32 kHz WAV."""
    raw_path = Path(raw_dir)
    proc_path = Path(processed_dir)

    for split in ["train", "val", "test"]:
        input_split = raw_path / split
        output_split = proc_path / split
        if not input_split.exists():
            continue

        files = list(input_split.glob("*.mp3")) + list(input_split.glob("*.wav"))
        print(f"\n⚙️ Normalizing {len(files)} files in '{split}' to 32 kHz mono...")

        success = 0
        for f in files:
            dest = output_split / f"{f.stem}.wav"
            if normalize_recording(f, dest):
                success += 1

        print(f"✓ Normalized {success}/{len(files)} files into {output_split}")


if __name__ == "__main__":
    normalize_all_audio()
