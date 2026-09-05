import os
import requests
import pandas as pd
from dotenv import load_dotenv

from audio import METADATA_PATH, ANNOTATIONS_PATH

load_dotenv()

API_URL = "https://xeno-canto.org/api/3/recordings"
API_KEY = os.getenv("XENO_CANTO_API_KEY")

QUERY = "cnt:nepal grp:birds"
records = []
annotations = []
page = 1


def get_metadata():
    global page
    while True:
        params = {
            "query": QUERY,
            "key": API_KEY,
            "page": page,
            "per_page": 100,
        }

        try:
            response = requests.get(API_URL, params=params)
            data = response.json()

            print("data", data)
            print(
                f"Page {page}/{data['numPages']} | Recordings: {data['numRecordings']}"
            )

            for r in data["recordings"]:
                # Recording metadata
                records.append(
                    {
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
                    }
                )

                # Annotation metadata
                annotation_set = r.get("annotation-set")

                if annotation_set:
                    for annotation in annotation_set.get("annotations", []):
                        annotations.append(
                            {
                                "recording_id": r["id"],
                                "scientific_name": annotation.get("scientific_name"),
                                "start_time": annotation.get("start_time"),
                                "end_time": annotation.get("end_time"),
                                "frequency_low": annotation.get("frequency_low"),
                                "frequency_high": annotation.get("frequency_high"),
                                "sound_type": annotation.get("sound_type"),
                                "sex": annotation.get("sex"),
                                "life_stage": annotation.get("life_stage"),
                                "remarks": annotation.get("annotation_remarks"),
                            }
                        )

            if page >= int(data["numPages"]):
                break

            page += 1

        except Exception as e:
            print("Error occured brother", e)
            raise e

    # Saving
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Saving recordings
    recordings_df = pd.DataFrame(records)
    recordings_df.to_csv(METADATA_PATH, index=False)

    print(f"\nSaved {len(recordings_df)} recordings.")

    # Saving annotations
    annotations_df = pd.DataFrame(annotations)
    annotations_df.to_csv(ANNOTATIONS_PATH, index=False)

    print(f"\nSaved {len(annotations_df)} annotations.")


if __name__ == "__main__":
    get_metadata()


"""
Saved 1568 recordings
Saved 0 annotations
"""
