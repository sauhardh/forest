import pandas as pd

from audio import METADATA_PATH, OUTPUT_PATH

MIN_SECONDS = 3


def filter_quality(df) -> pd.DataFrame:
    return df[df["quality"].isin(["A", "B", "C"])]


def filter_duration(df) -> pd.DataFrame:
    duration = pd.to_timedelta("00:" + df["duration"], errors="coerce")
    seconds = duration.dt.total_seconds()

    return df[seconds >= MIN_SECONDS]


# UNUSED
def find_counts(df) -> pd.DataFrame:
    counts = df["scientific_name"].value_counts()
    return counts


def metadata_filter():
    df = pd.read_csv(METADATA_PATH)

    print(f"Original recordings: {len(df)}")
    print(f"Original species: {df['scientific_name'].nunique()}")

    df = filter_quality(df)

    print(f"After quality filter: {len(df)}")

    df = filter_duration(df)

    print(f"After duration filter: {len(df)}")
    print(f"Species remaining: {df['scientific_name'].nunique()}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    metadata_filter()

"""
Original recordings: 1568
Original species: 311
After quality filter: 1529
After duration filter: 1523
Species remaining: 304
"""
