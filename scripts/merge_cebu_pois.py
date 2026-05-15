import os
import re
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(os.path.dirname(SCRIPT_DIR), "data")

def clean_name(name):
    if pd.isna(name):
        return ""
    name = str(name).lower().strip()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name

def best_match(query, candidates, threshold=65):
    query_clean = clean_name(query)
    cleaned_candidates = [clean_name(c) for c in candidates]
    result = process.extractOne(
        query_clean,
        cleaned_candidates,
        scorer=fuzz.WRatio,
        score_cutoff=threshold,
    )
    if result is None:
        return None, 0
    _match_str, score, idx = result
    return idx, round(score, 1)

print("Merge – Cebu POIs")

google = pd.read_csv(os.path.join(DATA_DIR, "google_places_cebu_pois.csv"))
ta     = pd.read_csv(os.path.join(DATA_DIR, "tripadvisor_cebu_pois.csv"))
osm    = pd.read_csv(os.path.join(DATA_DIR, "osm_cebu_pois.csv"))
sugbo  = pd.read_csv(os.path.join(DATA_DIR, "sugbo_cebu_pois.csv"))

print(f"loaded  google={len(google)}  ta={len(ta)}  osm={len(osm)}  sugbo={len(sugbo)}")

TYPES_MAP = {
    # specific parks before generic 'park'
    "amusement_park":      "entertainment",
    "water_park":          "entertainment",
    "city_park":           "nature",
    "wildlife_park":       "nature",
    "botanical_garden":    "nature",

    "beach":               "beach",
    "island":              "beach",
    "natural_feature":     "nature",
    "nature_preserve":     "nature",
    "park":                "nature",
    "garden":              "nature",
    "waterfall":           "nature",
    "viewpoint":           "nature",
    "scenic_spot":         "nature",
    "observation_deck":    "nature",
    "hiking_area":         "nature",
    "fishing_pond":        "nature",
    "campground":          "nature",

    "tourist_attraction":  "attraction",

    "sports_activity_location": "entertainment",
    "amusement_center":    "entertainment",
    "sports_complex":      "entertainment",
    "event_venue":         "entertainment",
    "stadium":             "entertainment",
    "swimming_pool":       "entertainment",
    "live_music_venue":    "entertainment",
    "night_club":          "entertainment",
    "go_karting_venue":    "entertainment",
    "race_course":         "entertainment",
    "adventure_sports_center": "entertainment",
    "art_studio":          "entertainment",
    "aquarium":            "entertainment",
    "zoo":                 "entertainment",

    # specific museum types before generic 'museum'
    "historical_place":    "history",
    "historical_landmark": "history",
    "history_museum":      "history",
    "monument":            "history",
    "landmark":            "history",
    "bridge":              "history",

    "art_museum":          "museum",
    "art_gallery":         "museum",
    "museum":              "museum",

    "church":              "religious",
    "buddhist_temple":     "religious",
    "hindu_temple":        "religious",
    "place_of_worship":    "religious",
    "cemetery":            "religious",

    "resort_hotel":        "accommodation",
    "hotel":               "accommodation",
    "lodging":             "accommodation",

    "european_restaurant": "food_drink",
    "coffee_shop":         "food_drink",
    "lounge_bar":          "food_drink",
    "food_court":          "food_drink",
    "food_store":          "food_drink",
    "restaurant":          "food_drink",
    "bakery":              "food_drink",
    "cafe":                "food_drink",
    "bar":                 "food_drink",
    "food":                "food_drink",

    "shopping_mall":       "shopping",
    "book_store":          "shopping",
    "market":              "shopping",
    "store":               "shopping",
}

SEARCH_QUERY_MAP = {
    "beaches":      "beach",
    "waterfalls":   "nature",
    "parks":        "nature",
    "gardens":      "nature",
    "viewpoints":   "nature",
    "religious":    "religious",
    "historical":   "history",
    "museums":      "museum",
    "islands":      "beach",
    "tourist":      "attraction",
    "things to do": "attraction",
}

def infer_category(row):
    cat = str(row.get("category", "")).strip().lower()
    if cat and cat != "nan":
        for key, val in TYPES_MAP.items():
            if key in cat:
                return val
    types = str(row.get("types", "")).lower()
    for key, val in TYPES_MAP.items():
        if key in types:
            return val
    sq = str(row.get("search_query", "")).lower()
    for key, val in SEARCH_QUERY_MAP.items():
        if key in sq:
            return val
    return "other"

google["category_norm"] = google.apply(infer_category, axis=1)

print("fuzzy matching TripAdvisor → Google …")

ta_names = ta["name"].tolist()
g_names  = google["name"].tolist()

ta_match_idx, ta_match_score, ta_match_name = [], [], []

for gname in g_names:
    idx, sc = best_match(gname, ta_names, threshold=70)
    ta_match_idx.append(idx)
    ta_match_score.append(sc)
    ta_match_name.append(ta.iloc[idx]["name"] if idx is not None else None)

matched_ta = sum(1 for s in ta_match_score if s > 0)
print(f"  ta matched: {matched_ta} / {len(google)}")

google["ta_matched_name"] = ta_match_name
google["ta_match_score"]  = ta_match_score
google["ta_rating"]       = [ta.iloc[i]["average_rating"] if i is not None else np.nan for i in ta_match_idx]
google["ta_review_count"] = [ta.iloc[i]["review_count"]   if i is not None else np.nan for i in ta_match_idx]
google["ta_category"]     = [ta.iloc[i]["category"]       if i is not None else None   for i in ta_match_idx]
google["ta_description"]  = [ta.iloc[i]["description"]    if i is not None else None   for i in ta_match_idx]
google["ta_url"]          = [ta.iloc[i]["url"]            if i is not None else None   for i in ta_match_idx]
google["ta_price_range"]  = [ta.iloc[i]["price_range"]    if i is not None else np.nan for i in ta_match_idx]

print("fuzzy matching OSM → Google …")

osm_names = osm["name"].tolist()

osm_match_idx, osm_match_score, osm_match_name = [], [], []

for _, grow in google.iterrows():
    idx, sc = best_match(grow["name"], osm_names, threshold=70)
    if idx is not None:
        osm_row  = osm.iloc[idx]
        lat_diff = abs(osm_row["latitude"]  - grow["latitude"])
        lon_diff = abs(osm_row["longitude"] - grow["longitude"])
        if lat_diff > 0.05 or lon_diff > 0.05:
            sc = sc * 0.7   # penalise geo mismatch
    osm_match_idx.append(idx if sc >= 70 else None)
    osm_match_score.append(sc if sc >= 70 else 0)
    osm_match_name.append(osm.iloc[idx]["name"] if (idx is not None and sc >= 70) else None)

matched_osm = sum(1 for s in osm_match_score if s > 0)
print(f"  osm matched: {matched_osm} / {len(google)}")

google["osm_matched_name"] = osm_match_name
google["osm_match_score"]  = osm_match_score
google["osm_category"]     = [osm.iloc[i]["category"]    if i is not None else None   for i in osm_match_idx]
google["osm_description"]  = [osm.iloc[i]["description"] if i is not None else None   for i in osm_match_idx]
google["osm_latitude"]     = [osm.iloc[i]["latitude"]    if i is not None else np.nan for i in osm_match_idx]
google["osm_longitude"]    = [osm.iloc[i]["longitude"]   if i is not None else np.nan for i in osm_match_idx]

print("fuzzy matching Sugbo → Google …")

sugbo_names = sugbo["name"].tolist()

sugbo_match_idx, sugbo_match_score, sugbo_match_name = [], [], []

for gname in g_names:
    idx, sc = best_match(gname, sugbo_names, threshold=65)
    sugbo_match_idx.append(idx)
    sugbo_match_score.append(sc)
    sugbo_match_name.append(sugbo.iloc[idx]["name"] if idx is not None else None)

matched_sugbo = sum(1 for s in sugbo_match_score if s > 0)
print(f"  sugbo matched: {matched_sugbo} / {len(google)}")

google["sugbo_matched_name"]   = sugbo_match_name
google["sugbo_match_score"]    = sugbo_match_score
google["sugbo_category"]       = [sugbo.iloc[i]["category"]       if i is not None else None for i in sugbo_match_idx]
google["sugbo_description"]    = [sugbo.iloc[i]["description"]    if i is not None else None for i in sugbo_match_idx]
google["sugbo_what_to_expect"] = [sugbo.iloc[i]["what_to_expect"] if i is not None else None for i in sugbo_match_idx]
google["sugbo_location_text"]  = [sugbo.iloc[i]["location_text"]  if i is not None else None for i in sugbo_match_idx]
google["sugbo_image_url"]      = [sugbo.iloc[i]["image_url"]      if i is not None else None for i in sugbo_match_idx]
google["sugbo_url"]            = [sugbo.iloc[i]["url"]            if i is not None else None for i in sugbo_match_idx]

merged = google.copy().rename(columns={
    "average_rating": "google_rating",
    "review_count":   "google_review_count",
    "description":    "google_description",
    "category":       "google_category_raw",
    "url":            "google_url",
})

merged.to_csv(os.path.join(DATA_DIR, "merged_cebu_pois.csv"), index=False)
print(f"saved merged_cebu_pois.csv  ({len(merged)} rows)")

drop_records = []
keep_mask    = pd.Series([True] * len(merged), index=merged.index)

no_rating = merged["google_rating"].isna()
for _, row in merged[no_rating].iterrows():
    drop_records.append({**row.to_dict(), "drop_reason": "no_google_rating"})
keep_mask[no_rating] = False
print(f"dropped {no_rating.sum()} rows (no rating)")

remaining = merged[keep_mask]
geo_bad   = (remaining["latitude"] > 11.5) | (remaining["longitude"] < 122)
for _, row in remaining[geo_bad].iterrows():
    drop_records.append({**row.to_dict(), "drop_reason": "geo_outlier_outside_cebu"})
keep_mask[geo_bad[geo_bad].index] = False
print(f"dropped {geo_bad.sum()} rows (geo outlier)")

dropped = pd.DataFrame(drop_records)
dropped.to_csv(os.path.join(DATA_DIR, "dropped_cebu_pois.csv"), index=False)

clean_base = merged[keep_mask].reset_index(drop=True)
print(f"remaining: {len(clean_base)} rows")

def cascade_description(row):
    for col in ["google_description", "sugbo_description", "ta_description", "osm_description"]:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            return str(val).strip()
    wte = row.get("sugbo_what_to_expect", "")
    if pd.notna(wte) and str(wte).strip():
        return str(wte).strip()
    cat  = str(row.get("category_norm", "place")).replace("_", " ")
    name = str(row.get("name", "This place"))
    return f"{name} is a {cat} located in Cebu, Philippines."

clean_base["description"] = clean_base.apply(cascade_description, axis=1)

clean = pd.DataFrame({
    "name":            clean_base["name"],
    "google_place_id": clean_base["google_place_id"],
    "search_query":    clean_base["search_query"],
    "latitude":        clean_base["latitude"],
    "longitude":       clean_base["longitude"],
    "address":         clean_base["address"],
    "average_rating":  clean_base["google_rating"],
    "review_count":    clean_base["google_review_count"],
    "category":        clean_base["category_norm"],
    "description":     clean_base["description"],
})

assert clean["category"].isna().sum() == 0
assert clean["description"].isna().sum() == 0
assert clean["average_rating"].isna().sum() == 0

clean.to_csv(os.path.join(DATA_DIR, "clean_merged_cebu_pois.csv"), index=False)
print(f"saved clean_merged_cebu_pois.csv ({len(clean)} rows)")

audit = pd.DataFrame({
    "name":                   clean_base["name"],
    "google_place_id":        clean_base["google_place_id"],
    "google_rating":          clean_base["google_rating"],
    "google_review_count":    clean_base["google_review_count"],
    "ta_rating":              clean_base["ta_rating"],
    "ta_review_count":        clean_base["ta_review_count"],
    "ta_matched_name":        clean_base["ta_matched_name"],
    "ta_match_score":         clean_base["ta_match_score"],
    "osm_matched_name":       clean_base["osm_matched_name"],
    "osm_match_score":        clean_base["osm_match_score"],
    "sugbo_matched_name":     clean_base["sugbo_matched_name"],
    "sugbo_match_score":      clean_base["sugbo_match_score"],
    "google_category_raw":    clean_base["google_category_raw"],
    "category_norm":          clean_base["category_norm"],
    "ta_category":            clean_base["ta_category"],
    "osm_category":           clean_base["osm_category"],
    "sugbo_category":         clean_base["sugbo_category"],
    "price_level_raw":        clean_base["price_level_raw"],
    "price_range":            clean_base["price_range"],
    "ta_price_range":         clean_base["ta_price_range"],
    "types":                  clean_base["types"],
    "website":                clean_base["website"],
    "google_url":             clean_base["google_url"],
    "ta_url":                 clean_base["ta_url"],
    "sugbo_url":              clean_base["sugbo_url"],
    "sugbo_what_to_expect":   clean_base["sugbo_what_to_expect"],
    "sugbo_image_url":        clean_base["sugbo_image_url"],
    "sugbo_location_text":    clean_base["sugbo_location_text"],
    "sample_reviews":         clean_base["sample_reviews"],
    "sample_review_ratings":  clean_base["sample_review_ratings"],
    "serves_breakfast":       clean_base["serves_breakfast"],
    "serves_lunch":           clean_base["serves_lunch"],
    "serves_dinner":          clean_base["serves_dinner"],
    "serves_coffee":          clean_base["serves_coffee"],
    "serves_dessert":         clean_base["serves_dessert"],
    "serves_vegetarian_food": clean_base["serves_vegetarian_food"],
    "serves_beer":            clean_base["serves_beer"],
    "serves_wine":            clean_base["serves_wine"],
    "takeout":                clean_base["takeout"],
    "dine_in":                clean_base["dine_in"],
    "delivery":               clean_base["delivery"],
    "outdoor_seating":        clean_base["outdoor_seating"],
    "osm_latitude":           clean_base["osm_latitude"],
    "osm_longitude":          clean_base["osm_longitude"],
})

audit.to_csv(os.path.join(DATA_DIR, "merge_audit_cebu_pois.csv"), index=False)
print(f"saved merge_audit_cebu_pois.csv ({len(audit)} rows)")
