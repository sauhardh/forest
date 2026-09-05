"""
Calculates NDVI = Normalized Difference Vegitation Index

NDVI = NIR - Red / (NIR + Red)
"""

import ee
import pandas as pd

ee.Initialize(project="forest-505711")


def get_ndvi(lat, lon, date, buffer_m=100):
    # buffer_m = 100; 100 meter radius area
    point = ee.Geometry.Point([lon, lat]).buffer(buffer_m)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate(ee.Date(date).advance(-15, "day"), ee.Date(date).advance(15, "day"))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    if collection.size().getInfo() == 0:
        return None

    img = collection.first()
    ndvi = img.normalizedDifference(["B8", "B4"])
    return ndvi.reduceRegion(ee.Reducer.mean(), point, 10).get("nd").getInfo()


if __name__ == "__main__":
    from pathlib import Path

    sites = pd.read_csv("data/sites.csv")
    sites["ndvi_label"] = sites.apply(
        lambda r: get_ndvi(r["lat"], r["lon"], r["recording_date"]), axis=1
    )

    Path("outputs").mkdir(parents=True, exist_ok=True)
    sites.to_csv("outputs/ndvi_labels.csv", index=False)
    print(sites)
