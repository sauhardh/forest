import sys
from pathlib import Path

# Ensure 'app' directory is in sys.path
_current = Path(__file__).resolve()
for _p in [_current.parent, _current.parent.parent, _current.parent.parent.parent]:
    if _p.name == "app" and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
    elif (_p / "app").is_dir() and str(_p / "app") not in sys.path:
        sys.path.insert(0, str(_p / "app"))

import pandas as pd
import requests

from audio import SPLIT_DIR, AUDIO_DIR


SPLITS = ["train", "test", "val"]


def download_audio(split: str):
    csv_path = Path(SPLIT_DIR / f"{split}.csv")
    output_dir = Path(AUDIO_DIR / split)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"\nDownloading {split}: {len(df)} recordings")

    for _, row in df.iterrows():
        recording_id = str(row["id"])
        audio_url = row["audio_url"]

        # Skip rows with missing audio URLs
        if not isinstance(audio_url, str) or not audio_url.strip():
            print(f"Skipping XC{recording_id}: missing audio URL")
            continue

        # Skip malformed URLs
        audio_url = str(audio_url).strip()
        if not audio_url.startswith(("http://", "https://")):
            print(f"Skipping XC{recording_id}: invalid audio URL {audio_url!r}")
            continue

        output_path = output_dir / f"XC{recording_id}.mp3"

        if output_path.exists():
            print(f"Already exists: XC{recording_id}")
            continue

        print(f"Downloading XC{recording_id}...")

        try:
            response = requests.get(audio_url, timeout=120)
            response.raise_for_status()
            output_path.write_bytes(response.content)

        except requests.RequestException as e:
            print(f"Failed XC{recording_id}: ", e)

    print(f"Finished {split}")


def main() -> None:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        download_audio(split)


if __name__ == "__main__":
    main()
