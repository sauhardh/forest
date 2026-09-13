"""
Step 2: Quality Filtering and Leak-Free Dataset Splitting.
Filters recordings by quality grade ('A' and 'B'), enforces duration bounds,
and splits into Train (80%), Val (10%), Test (10%) at the recording level.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def filter_and_split_dataset(
    input_csv: str = "data/metadata_raw.csv",
    output_dir: str = "data/splits",
    min_recordings_per_species: int = 3,
):
    """Filters metadata by quality and produces leak-free train/val/test splits."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    print(f"Total raw recordings: {len(df)} across {df['scientific_name'].nunique()} species")

    # 1. Quality Filter: Keep only high-SNR recordings ('A' and 'B' grades)
    df = df[df["quality"].isin(["A", "B", "a", "b"])].copy()

    # 2. Duration Filter: Must be at least 3.0 seconds (one analysis window)
    df["length_seconds"] = pd.to_numeric(df["length_seconds"], errors="coerce").fillna(0)
    df = df[(df["length_seconds"] >= 3.0) & (df["length_seconds"] <= 300.0)].copy()

    # 3. Minimum Recordings Filter: Ensure species have enough samples to appear in train and eval
    species_counts = df["scientific_name"].value_counts()
    valid_species = species_counts[species_counts >= min_recordings_per_species].index
    df = df[df["scientific_name"].isin(valid_species)].copy()

    print(f"Filtered clean recordings: {len(df)} across {df['scientific_name'].nunique()} species")

    # 4. Stratified Recording-Level Split (80% Train, 10% Val, 10% Test)
    # Grouping by species prevents single-recording leakage between train and test.
    train_list, val_list, test_list = [], [], []

    np.random.seed(42)
    for species, grp in df.groupby("scientific_name"):
        shuffled = grp.sample(frac=1.0, random_state=42).reset_index(drop=True)
        n = len(shuffled)
        if n == 3:
            train_list.append(shuffled.iloc[[0]])
            val_list.append(shuffled.iloc[[1]])
            test_list.append(shuffled.iloc[[2]])
        else:
            n_test = max(1, int(0.10 * n))
            n_val = max(1, int(0.10 * n))
            n_train = n - n_val - n_test

            train_list.append(shuffled.iloc[:n_train])
            val_list.append(shuffled.iloc[n_train : n_train + n_val])
            test_list.append(shuffled.iloc[n_train + n_val :])

    train_df = pd.concat(train_list).reset_index(drop=True)
    val_df = pd.concat(val_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    train_df.to_csv(out_path / "train.csv", index=False)
    val_df.to_csv(out_path / "val.csv", index=False)
    test_df.to_csv(out_path / "test.csv", index=False)

    print(f"✓ Splits created: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")


if __name__ == "__main__":
    filter_and_split_dataset()
