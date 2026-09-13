"""
Step 3: Concurrent Audio Downloader.
Downloads MP3 recordings from Xeno-Canto CDN into split subfolders (data/raw/train, val, test).
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd


def download_single_file(url: str, output_path: Path) -> bool:
    """Downloads an audio stream and writes to disk."""
    if output_path.exists() and output_path.stat().st_size > 1000:
        return True  # Already downloaded

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(resp.content)
            return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
    return False


def download_dataset_audio(splits_dir: str = "data/splits", raw_dir: str = "data/raw", max_workers: int = 8):
    """Concurrently downloads audio files for all dataset splits."""
    splits_path = Path(splits_dir)
    raw_path = Path(raw_dir)

    for split in ["train", "val", "test"]:
        csv_file = splits_path / f"{split}.csv"
        if not csv_file.exists():
            continue

        df = pd.read_csv(csv_file)
        split_dir = raw_path / split
        split_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📥 Downloading {len(df)} recordings for '{split}'...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for _, row in df.iterrows():
                url = row["audio_url"]
                rec_id = str(row["recording_id"])
                dest = split_dir / f"XC{rec_id}.mp3"
                futures[executor.submit(download_single_file, url, dest)] = rec_id

            success = 0
            for future in as_completed(futures):
                if future.result():
                    success += 1

            print(f"✓ '{split}' download complete: {success}/{len(df)} files available")


if __name__ == "__main__":
    download_dataset_audio()
