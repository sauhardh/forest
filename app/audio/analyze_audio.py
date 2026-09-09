import sys
from pathlib import Path

# Ensure 'app' directory is in sys.path
_current = Path(__file__).resolve()
for _p in [_current.parent, _current.parent.parent, _current.parent.parent.parent]:
    if _p.name == "app" and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
    elif (_p / "app").is_dir() and str(_p / "app") not in sys.path:
        sys.path.insert(0, str(_p / "app"))

import numpy as np
import pandas as pd

import librosa
from audio import ANALYZE_PATH, PROCESSED_DIR



def analyze_file(path: Path) -> dict:
    y, sr = librosa.load(path, sr=32_000, mono=True)

    rms = librosa.feature.rms(y=y)[0]

    return {
        "id": path.stem,
        "duration": len(y) / sr,
        "mean_rms": float(np.mean(rms)),
        "max_rms": float(np.max(rms)),
        "samples": len(y),
    }


def main() -> None:
    results = []

    for split in ["train", "test", "val"]:
        audio_dir = PROCESSED_DIR / split

        for path in audio_dir.glob("*.wav"):
            print("path", path)
            result = analyze_file(path)
            result["split"] = split

            results.append(result)

    df = pd.DataFrame(results)
    ANALYZE_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(ANALYZE_PATH, index=False)

    print(df.describe())
    print(
        "\nSaved:",
        ANALYZE_PATH,
    )


if __name__ == "__main__":
    main()

"""
duration     mean_rms      max_rms       samples
count  1498.000000  1498.000000  1498.000000  1.498000e+03
mean     52.336210     0.028787     0.208182  1.674759e+06
std      64.483423     0.025745     0.131856  2.063470e+06
min       2.951844     0.000693     0.001705  9.445900e+04
25%      19.833492     0.012731     0.111161  6.346718e+05
50%      36.088891     0.021373     0.183092  1.154844e+06
75%      62.617383     0.036146     0.283930  2.003756e+06
max    1395.735969     0.232844     0.750780  4.466355e+07
"""
