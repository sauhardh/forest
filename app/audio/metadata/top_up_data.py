"""
Top-up data for under-represented species.

Identifies species with fewer than MIN_RECORDINGS_TARGET recordings and fetches
more from XenoCanto using neighboring countries (India, Bhutan, China).

Usage:
    uv run python3 -m audio.metadata.top_up_data            # dry-run
    uv run python3 -m audio.metadata.top_up_data --apply    # actually fetch

After --apply, re-run the full pipeline:
    uv run python3 -m audio.split_dataset
    uv run python3 -m audio.download_audio
    uv run python3 -m audio.process.normalize_audio
    uv run python3 -m audio.process.extract_clips
"""

import sys, os, time, argparse
from pathlib import Path

_current = Path(__file__).resolve()
for _p in [_current.parent, _current.parent.parent, _current.parent.parent.parent]:
    if _p.name == "app" and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
    elif (_p / "app").is_dir() and str(_p / "app") not in sys.path:
        sys.path.insert(0, str(_p / "app"))

import requests
import pandas as pd
from dotenv import load_dotenv
from audio import OUTPUT_PATH

load_dotenv()

API_URL = "https://xeno-canto.org/api/3/recordings"
API_KEY = os.getenv("XENO_CANTO_API_KEY")

MIN_RECORDINGS_TARGET = 8  # Conservative target: guarantees ~5-6 train, 1 val, 1 test
EXTRA_COUNTRIES = ["bhutan", "india"]  # Closest ecological neighbors
ACCEPTED_QUALITY = {"A", "B", "C"}
MIN_SECONDS = 3
MAX_PER_COUNTRY = 8
API_SLEEP = 0.5


def duration_to_seconds(s):
    try:
        parts = [float(p) for p in str(s).strip().split(":")]
        if len(parts) == 2: return parts[0]*60 + parts[1]
        if len(parts) == 3: return parts[0]*3600 + parts[1]*60 + parts[2]
    except: pass
    return 0.0


def fetch_recordings(species_name, country, max_results):
    parts = species_name.strip().split()
    if len(parts) >= 2:
        query = f"gen:{parts[0]} sp:{parts[1]} cnt:{country}"
    elif len(parts) == 1:
        query = f"gen:{parts[0]} cnt:{country}"
    else:
        return []

    results, page = [], 1
    while len(results) < max_results:
        params = {"query": query, "page": page}
        if API_KEY:
            params["key"] = API_KEY
        try:
            resp = requests.get(API_URL, params=params, timeout=30)
            data = resp.json()
            if "error" in data:
                print(f"    [API error] {data.get('message')}")
                break
        except Exception as e:
            print(f"    [Request error] {e}")
            break

        for r in data.get("recordings", []):
            results.append({
                "id": r["id"],
                "scientific_name": f"{r['gen']} {r['sp']}",
                "english_name": r.get("en"),
                "type": r.get("type"),
                "quality": r.get("q"),
                "duration": r.get("length"),
                "sample_rate": r.get("smp"),
                "latitude": r.get("lat"),
                "longitude": r.get("lon"),
                "date": r.get("date"),
                "audio_url": r.get("file"),
                "license": r.get("lic"),
            })
            if len(results) >= max_results:
                break
        if page >= int(data.get("numPages", 1)) or len(results) >= max_results:
            break
        page += 1
        time.sleep(API_SLEEP)
    return results


def filter_recordings(records, existing_ids):
    kept = []
    for r in records:
        rid = str(r["id"])
        if rid in existing_ids: continue
        if r.get("quality") not in ACCEPTED_QUALITY: continue
        if duration_to_seconds(r.get("duration","0:00")) < MIN_SECONDS: continue
        kept.append(r); existing_ids.add(rid)
    return kept


def main(apply=False, min_target=MIN_RECORDINGS_TARGET):
    if not OUTPUT_PATH.exists():
        print(f"ERROR: {OUTPUT_PATH} not found. Run metadata_filter first.")
        sys.exit(1)

    df = pd.read_csv(OUTPUT_PATH)
    existing_ids = set(df["id"].astype(str))
    counts = df["scientific_name"].value_counts()
    starved = counts[counts < min_target].sort_values()

    print(f"Current dataset : {len(df)} recordings | {counts.shape[0]} species")
    print(f"Target          : {min_target} recordings/species")
    print(f"Under target    : {len(starved)} species\n")
    print(f"{'Species':<42} {'Have':>5} {'Need':>5}")
    print("-" * 55)
    for sp, n in starved.items():
        print(f"  {sp:<40} {n:>5} {min_target-n:>5}")
    print()

    if not apply:
        print("DRY RUN - no changes made. Use --apply to fetch recordings.\n")
        return

    new_rows = []
    for species, current_count in starved.items():
        needed = min_target - current_count
        print(f"\n-> {species}  (have {current_count}, need {needed} more)")
        fetched = []
        for country in EXTRA_COUNTRIES:
            if len(fetched) >= needed: break
            recs = fetch_recordings(species, country, MAX_PER_COUNTRY)
            good = filter_recordings(recs, existing_ids)
            print(f"   {country:8s}: {len(recs):3d} found -> {len(good):3d} new")
            fetched.extend(good)
            time.sleep(API_SLEEP)
        take = fetched[:needed]
        print(f"   -> adding {len(take)} recordings")
        new_rows.extend(take)

    if not new_rows:
        print("\nNo new recordings found. Dataset unchanged.")
        return

    combined = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\nAdded {len(new_rows)} recordings: {len(df)} -> {len(combined)}")
    print(f"Saved: {OUTPUT_PATH}")
    print("\nNext steps:")
    print("  uv run python3 -m audio.split_dataset")
    print("  uv run python3 -m audio.download_audio")
    print("  uv run python3 -m audio.process.normalize_audio")
    print("  uv run python3 -m audio.process.extract_clips")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--min-recordings", type=int, default=MIN_RECORDINGS_TARGET)
    args = parser.parse_args()
    main(apply=args.apply, min_target=args.min_recordings)
