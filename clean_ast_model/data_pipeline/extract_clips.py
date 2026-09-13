"""
Step 5: Bird Activity Detection and 3.0-Second Clip Extraction.
Uses a 500 Hz - 15 kHz Butterworth bandpass filter and RMS energy thresholding
to extract standardized 3.0-second vocalization clips and builds clips_metadata.csv.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy.signal import butter, sosfilt

SAMPLE_RATE = 32000
CLIP_DURATION = 3.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)   # 96,000 samples
STRIDE_SAMPLES = int(1.0 * SAMPLE_RATE)           # 1.0 second sliding window stride


class BirdActivityDetector:
    """Filters out low-frequency rumble and detects active bird call segments."""
    def __init__(self, low_hz: float = 500.0, high_hz: float = 14000.0, k: float = 1.2):
        nyquist = SAMPLE_RATE / 2.0
        self.sos = butter(4, [low_hz / nyquist, high_hz / nyquist], btype="band", output="sos")
        self.k = k

    def detect(self, audio: np.ndarray) -> np.ndarray:
        """Returns boolean mask where True indicates bird vocalization activity."""
        filtered = sosfilt(self.sos, audio)
        rms = librosa.feature.rms(y=filtered, frame_length=1024, hop_length=512)[0]
        threshold = np.median(rms) + self.k * np.std(rms)
        return rms > threshold


def extract_clips_from_dataset(
    processed_dir: str = "data/processed",
    splits_dir: str = "data/splits",
    output_clips_dir: str = "data/clips",
    output_csv: str = "data/clips_metadata.csv",
):
    """Processes normalized WAVs into uniform 3.0s clips and writes master metadata CSV."""
    proc_path = Path(processed_dir)
    splits_path = Path(splits_dir)
    clips_path = Path(output_clips_dir)
    detector = BirdActivityDetector()

    records = []

    for split in ["train", "val", "test"]:
        split_csv = splits_path / f"{split}.csv"
        if not split_csv.exists():
            continue

        meta_df = pd.read_csv(split_csv)
        rec_to_species = dict(zip(meta_df["recording_id"].astype(str), meta_df["scientific_name"]))

        split_audio_dir = proc_path / split
        wav_files = list(split_audio_dir.glob("*.wav"))
        print(f"\n✂️ Extracting clips from '{split}' ({len(wav_files)} recordings)...")

        for wav_path in wav_files:
            rec_id = wav_path.stem.replace("XC", "")
            species = rec_to_species.get(rec_id, "unknown")
            species_clean = species.replace(" ", "_")

            try:
                audio, sr = sf.read(str(wav_path), dtype="float32")
                if len(audio) < CLIP_SAMPLES:
                    continue

                active_mask = detector.detect(audio)
                n_frames = len(active_mask)

                # Slide a 3.0s window across the recording
                start = 0
                clip_idx = 0
                while start + CLIP_SAMPLES <= len(audio):
                    frame_start = int((start / len(audio)) * n_frames)
                    frame_end = int(((start + CLIP_SAMPLES) / len(audio)) * n_frames)

                    # Only keep window if at least 25% contains active bird call energy
                    if active_mask[frame_start:frame_end].mean() > 0.25:
                        clip_data = audio[start : start + CLIP_SAMPLES]
                        sec_offset = start / SAMPLE_RATE

                        species_dir = clips_path / split / species_clean
                        species_dir.mkdir(parents=True, exist_ok=True)

                        clip_file = species_dir / f"XC{rec_id}_{sec_offset:.2f}.wav"
                        sf.write(str(clip_file), clip_data, SAMPLE_RATE)

                        records.append({
                            "recording_id": rec_id,
                            "clip_path": str(clip_file.resolve()),
                            "species": species,
                            "split": split,
                            "start_sec": sec_offset,
                            "end_sec": sec_offset + CLIP_DURATION,
                        })
                        clip_idx += 1

                    start += STRIDE_SAMPLES

            except Exception as e:
                print(f"Error extracting {wav_path.name}: {e}")

    df_out = pd.DataFrame(records)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"\n✓ Master clips table created: {len(df_out)} clips across {df_out['species'].nunique()} species saved to {output_path}")


if __name__ == "__main__":
    extract_clips_from_dataset()
