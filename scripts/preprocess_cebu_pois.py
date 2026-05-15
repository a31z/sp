import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(os.path.dirname(SCRIPT_DIR), "data")

df = pd.read_csv(os.path.join(DATA_DIR, "clean_merged_cebu_pois.csv"))
print(f"loaded {df.shape[0]} rows × {df.shape[1]} cols")

df["rating_norm"] = (df["average_rating"] - 1) / 4.0

df["review_count_log"]  = np.log1p(df["review_count"])
scaler                  = MinMaxScaler()
df["review_count_norm"] = scaler.fit_transform(df[["review_count_log"]]).flatten()

# pulls low-review POIs toward the global mean
m = 50  # a POI needs ~50 reviews before its rating is fully trusted
C = df["rating_norm"].mean()

df["weighted_score"] = (
    (df["review_count"] / (df["review_count"] + m)) * df["rating_norm"]
    + (m / (df["review_count"] + m)) * C
)

def assign_zone(lat, lon):
    if lat > 10.60:  return "north_cebu"
    if lat < 9.90:   return "south_cebu"
    if lat > 10.40:  return "north_metro"
    if lat < 10.25:  return "south_metro"
    if lon > 123.92: return "eastern_cebu"
    return "cebu_city"

df["location_zone"] = df.apply(lambda r: assign_zone(r["latitude"], r["longitude"]), axis=1)

le_cat  = LabelEncoder()
le_zone = LabelEncoder()

df["category_enc"] = le_cat.fit_transform(df["category"])
df["zone_enc"]     = le_zone.fit_transform(df["location_zone"])

cat_dummies  = pd.get_dummies(df["category"],      prefix="cat")
zone_dummies = pd.get_dummies(df["location_zone"], prefix="zone")

encoder_map = {
    "category": {
        label: int(code)
        for label, code in zip(le_cat.classes_, le_cat.transform(le_cat.classes_))
    },
    "location_zone": {
        label: int(code)
        for label, code in zip(le_zone.classes_, le_zone.transform(le_zone.classes_))
    },
}
with open(os.path.join(DATA_DIR, "encoder_map.json"), "w") as f:
    json.dump(encoder_map, f, indent=2)

feature_cols = [
    "rating_norm",
    "review_count_norm",
    "weighted_score",
    "category_enc",
    "zone_enc",
]

poi_features = pd.concat(
    [
        df[["google_place_id", "name", "category", "location_zone"] + feature_cols],
        cat_dummies,
        zone_dummies,
    ],
    axis=1,
)

poi_features.to_csv(os.path.join(DATA_DIR, "poi_features.csv"), index=False)
print(f"saved poi_features.csv  ({poi_features.shape[0]} rows × {poi_features.shape[1]} cols)")
print(f"saved encoder_map.json")
