import librosa
import numpy as np
import pandas as pd

import soundfile as sf
from scipy.signal import butter, sosfilt

from audio import SAMPLE_RATE, CLIPS_DIR, CLIPS_METADATA_PATH, PROCESSED_DIR, SPLIT_DIR


LOW_HZ = 500
HIGH_HZ = 15_000
HOP_SAMPLES = 1 * SAMPLE_RATE  # 1-second stride
FRAME_LEN = 1024  # 32 ms
HOP_LENGTH = 768  # 24 ms # overlap with Frame = 25%
CLIP_SAMPLES = 3 * SAMPLE_RATE  # 96,000 samples
MIN_ACTIVE_FRAMES = int(
    0.35 * SAMPLE_RATE / HOP_LENGTH
)  # window must have ≥ 0.35 s of bird sound


class BirdActivityDetector:
    def __init__(self, k: float = 1.5):
        nyq = SAMPLE_RATE / 2
        self._sos = butter(4, [LOW_HZ / nyq, HIGH_HZ / nyq], btype="band", output="sos")
        self.k = k  # k is threshold multiplier

    def detect(self, audio: np.ndarray):
        filtered = sosfilt(self._sos, audio)
        rms = librosa.feature.rms(
            y=filtered, frame_length=FRAME_LEN, hop_length=HOP_LENGTH
        )[0]

        threshold = np.median(rms) + self.k * rms.std()
        mask = rms > threshold
        peak_sample = int(np.argmax(rms)) * HOP_LENGTH

        return mask, peak_sample


class ExtractClips:
    def __init__(self):
        pass

    def _active_count(self, mask: np.ndarray, win_start: int) -> int:
        f0 = win_start // HOP_LENGTH
        f1 = min(len(mask), f0 + CLIP_SAMPLES // HOP_LENGTH)
        return int(mask[f0:f1].sum())

    def _clip_path(self, recording_id: str, species: str, split: str, start_sec: float):
        path = (
            CLIPS_DIR
            / split
            / species.replace(" ", "_")
            / f"{recording_id}_{start_sec:.2f}.wav"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _row(
        self, recording_id: str, species: str, split: str, start_sec: float, path
    ) -> dict:
        return {
            "clip_path": str(path),
            "recording_id": recording_id,
            "species": species,
            "split": split,
            "start_sec": round(start_sec, 4),
            "end_sec": round(start_sec + 3.0, 4),
        }

    def extract_clips(
        self,
        audio: np.ndarray,
        mask: np.ndarray,
        peak_sample: int,
        recording_id: str,
        species: str,
        split: str,
    ):
        n = len(audio)
        rows = []

        # Short recording: pad to 3 s, centre on loudest moment
        if n < CLIP_SAMPLES:
            start = max(0, peak_sample - CLIP_SAMPLES // 2)
            clip = np.zeros(CLIP_SAMPLES, dtype=np.float32)
            clip[: n - start] = audio[start:]

            if self._active_count(mask, start) >= MIN_ACTIVE_FRAMES:
                path = self._clip_path(
                    recording_id, species, split, start / SAMPLE_RATE
                )
                sf.write(path, clip, SAMPLE_RATE)
                rows.append(
                    self._row(recording_id, species, split, start / SAMPLE_RATE, path)
                )

            return rows

        # Long recording: sliding 3-second window, 1-second hop
        for win_start in range(0, n - CLIP_SAMPLES + 1, HOP_SAMPLES):
            if self._active_count(mask, win_start) < MIN_ACTIVE_FRAMES:
                continue
            start_sec = win_start / SAMPLE_RATE
            path = self._clip_path(recording_id, species, split, start_sec)
            sf.write(path, audio[win_start : win_start + CLIP_SAMPLES], SAMPLE_RATE)
            rows.append(self._row(recording_id, species, split, start_sec, path))
        return rows


def main() -> None:
    detector = BirdActivityDetector()
    extractor = ExtractClips()
    all_rows = []

    for split in ["train", "val", "test"]:
        meta = pd.read_csv(SPLIT_DIR / f"{split}.csv")[["id", "scientific_name"]]
        id_to_species: dict[str, str] = {
            f"XC{r['id']}": str(r["scientific_name"]) for _, r in meta.iterrows()
        }
        wav_files = sorted((PROCESSED_DIR / split).glob("*.wav"))
        print(f"\n{split}: {len(wav_files)} files")

        for path in wav_files:
            rid = path.stem
            species = id_to_species.get(rid)
            if species is None:
                print(f"  [skip] {rid}: no metadata")
                continue
            audio, _ = sf.read(path, dtype="float32")
            mask, peak_sample = detector.detect(audio)
            if not mask.any():
                print(f"  [skip] {rid}: no bird activity")
                continue
            rows = extractor.extract_clips(
                audio, mask, peak_sample, rid, species, split
            )
            all_rows.extend(rows)
            print(f"  {rid} → {len(rows)} clips")

    df = pd.DataFrame(all_rows)
    CLIPS_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLIPS_METADATA_PATH, index=False)
    print(f"\nTotal clips: {len(df)}  →  {CLIPS_METADATA_PATH}")


if __name__ == "__main__":
    main()
