"""
Step 1: Download bird audio recording metadata from the Xeno-Canto API.
Retrieves recording IDs, species scientific names, quality ratings, audio lengths, and download URLs.
"""

from pathlib import Path
import requests
import pandas as pd

API_URL = "https://xeno-canto.org/api/3/recordings"


def fetch_xeno_canto_metadata(query: str = "cnt:nepal grp:birds", output_csv: str = "data/metadata_raw.csv"):
    """Queries Xeno-Canto API page-by-page and saves raw metadata table."""
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    page = 1

    print(f"📡 Querying Xeno-Canto API for: '{query}'...")
    while True:
        response = requests.get(API_URL, params={"query": query, "page": page})
        if response.status_code != 200:
            print(f"Failed to fetch page {page}: {response.status_code}")
            break

        data = response.json()
        num_pages = int(data.get("numPages", 1))
        recordings = data.get("recordings", [])

        print(f"  Page [{page:02d}/{num_pages:02d}] | Fetched {len(recordings)} recordings")

        for r in recordings:
            records.append({
                "recording_id": r.get("id"),
                "scientific_name": f"{r.get('gen', '')} {r.get('sp', '')}".strip(),
                "english_name": r.get("en", ""),
                "quality": r.get("q", ""),           # A (highest) to E (lowest)
                "length_seconds": r.get("length", 0),
                "audio_url": r.get("file", ""),
                "country": r.get("cnt", ""),
                "license": r.get("lic", ""),
            })

        if page >= num_pages:
            break
        page += 1

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"✓ Saved {len(df)} recordings across {df['scientific_name'].nunique()} species to {output_path}")


if __name__ == "__main__":
    fetch_xeno_canto_metadata()
