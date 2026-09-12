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


MIN_RECORDINGS_PER_SPECIES = 3  # Need at least 3 recordings to have 1 in each split


def split_dataset():
    df = pd.read_csv(INPUT_PATH)

    # Remove accidental duplicate recording IDs
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    print(f"Total recordings before filtering: {len(df)}")
    print(f"Total species before filtering: {df['scientific_name'].nunique()}")

    # Drop species with too few recordings to stratify across all 3 splits.
    # sklearn's train_test_split requires at least 2 samples per class to stratify.
    # We need at least 3 to ensure representation in train + val + test.
    counts = df["scientific_name"].value_counts()
    keep = counts[counts >= MIN_RECORDINGS_PER_SPECIES].index
    dropped = counts[counts < MIN_RECORDINGS_PER_SPECIES]
    if len(dropped):
        print(f"\nDropping {len(dropped)} species with < {MIN_RECORDINGS_PER_SPECIES} recordings:")
        for sp, n in dropped.items():
            print(f"  {sp}: {n} recording(s)")
    df = df[df["scientific_name"].isin(keep)].reset_index(drop=True)

    print(f"\nTotal recordings after filtering: {len(df)}")
    print(f"Total species after filtering: {df['scientific_name'].nunique()}")

    # Shuffle per species and allocate: ~80% train, ~10% val, ~10% test.
    # Guaranteed: every species with >= 3 recordings gets >= 1 in train, 1 in val, 1 in test.
    train_parts, val_parts, test_parts = [], [], []

    for sp, group in df.groupby("scientific_name"):
        recs = group.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        n = len(recs)
        n_val = max(1, int(round(n * 0.10)))
        n_test = max(1, int(round(n * 0.10)))
        # Guard: ensure at least 1 recording stays in train
        if n - n_val - n_test < 1:
            n_val = 1
            n_test = 1
        n_train = n - n_val - n_test

        train_parts.append(recs.iloc[:n_train])
        val_parts.append(recs.iloc[n_train:n_train + n_val])
        test_parts.append(recs.iloc[n_train + n_val:])

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

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
