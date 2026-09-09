import sys
from pathlib import Path
from typing import cast

# Ensure 'app' directory is in sys.path
_current = Path(__file__).resolve()
for _p in [_current.parent, _current.parent.parent, _current.parent.parent.parent]:
    if _p.name == "app" and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
    elif (_p / "app").is_dir() and str(_p / "app") not in sys.path:
        sys.path.insert(0, str(_p / "app"))

import pandas as pd
from sklearn.model_selection import train_test_split

from audio import INPUT_PATH, SPLIT_DIR


RANDOM_STATE = 42


def split_dataset():
    df = pd.read_csv(INPUT_PATH)

    # Remove accidental duplicate recording IDs
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    print(f"Total recordings: {len(df)}")
    print(f"Total species: {df['scientific_name'].nunique()}")

    # Shuffle recordings.
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # 80% train, 20% temporary
    train_df, temp_df = cast(
        "tuple[pd.DataFrame, pd.DataFrame]",
        train_test_split(df, test_size=0.20, random_state=42),
    )

    # 10% validation, 10% test
    val_df, test_df = cast(
        "tuple[pd.DataFrame, pd.DataFrame]",
        train_test_split(temp_df, test_size=0.50, random_state=42),
    )

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(SPLIT_DIR / "train.csv", index=False)
    val_df.to_csv(SPLIT_DIR / "val.csv", index=False)
    test_df.to_csv(SPLIT_DIR / "test.csv", index=False)

    # Verify leakage
    train_ids = set(train_df["id"])
    val_ids = set(val_df["id"])
    test_ids = set(test_df["id"])

    print("\nLeakage check:")
    print("Train ∩ Val:", train_ids & val_ids)
    print("Train ∩ Test:", train_ids & test_ids)
    print("Val ∩ Test:", val_ids & test_ids)

    print("\nSplit sizes:")
    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))


if __name__ == "__main__":
    split_dataset()

"""
Total recordings: 1523
Total species: 304

Leakage check:
Train ∩ Val: set()
Train ∩ Test: set()
Val ∩ Test: set()

Split sizes:
Train: 1218
Validation: 152
Test: 153
"""
