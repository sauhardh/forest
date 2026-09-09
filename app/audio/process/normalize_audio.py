import sys
from pathlib import Path

# Ensure 'app' directory is in sys.path
_current = Path(__file__).resolve()
for _p in [_current.parent, _current.parent.parent, _current.parent.parent.parent]:
    if _p.name == "app" and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
    elif (_p / "app").is_dir() and str(_p / "app") not in sys.path:
        sys.path.insert(0, str(_p / "app"))

import librosa
import soundfile as sf

from audio import RAW_DIR, PROCESSED_DIR, SAMPLE_RATE



def normalize_file(input_path: Path, output_path: Path):
    try:
        audio, _ = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)

        sf.write(output_path, audio, SAMPLE_RATE)
        print(f"✓ {input_path.name}")

    except Exception as e:
        print(f"✗ {input_path}: {e}")


def main() -> None:
    for split in ["train", "test", "val"]:
        input_dir = RAW_DIR / split
        output_dir = PROCESSED_DIR / split

        output_dir.mkdir(parents=True, exist_ok=True)

        files = list(input_dir.glob("*.mp3"))
        print(f"\n{split}: {len(files)} MP3 files")

        for input_path in files:
            output_path = output_dir / f"{input_path.stem}.wav"

            if output_path.exists():
                print(f"Already exists: {output_path.name}")
                continue

            normalize_file(
                input_path,
                output_path,
            )


if __name__ == "__main__":
    main()
