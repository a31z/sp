import os
import json
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CBF_DIR    = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(os.path.dirname(CBF_DIR), "data")
MODELS_DIR = os.path.join(CBF_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MAX_FEATURES = None
NGRAM_RANGE  = (1, 2)
MIN_DF       = 2
STOP_WORDS   = "english"
TEXT_WEIGHT  = 0.5  # β: 1.0 = pure text, 0.0 = pure attributes

clean = pd.read_csv(os.path.join(DATA_DIR, "clean_merged_cebu_pois_v2.csv"))
feats = pd.read_csv(os.path.join(DATA_DIR, "poi_features.csv"))

clean = clean.set_index("google_place_id")
feats = feats.set_index("google_place_id")
common_ids = clean.index.intersection(feats.index)
clean = clean.loc[common_ids].reset_index()
feats = feats.loc[common_ids].reset_index()

assert len(clean) == len(feats), "row mismatch between clean and feats"
print(f"loaded {len(clean)} POIs")

vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=NGRAM_RANGE,
    min_df=MIN_DF,
    stop_words=STOP_WORDS,
    lowercase=True,
    strip_accents="unicode",
)
text_matrix = vectorizer.fit_transform(clean["description"].fillna(""))
text_matrix = normalize(text_matrix, norm="l2", axis=1)
print(f"tfidf vocab: {len(vectorizer.vocabulary_)} terms  matrix: {text_matrix.shape}")

attr_cols = (
    [c for c in feats.columns if c.startswith("cat_")]
  + [c for c in feats.columns if c.startswith("zone_") and c != "zone_enc"]
)
attr_matrix = feats[attr_cols].astype(float).values
attr_matrix = normalize(attr_matrix, norm="l2", axis=1)

text_weighted = text_matrix * TEXT_WEIGHT
attr_weighted = sparse.csr_matrix(attr_matrix * (1 - TEXT_WEIGHT))
poi_matrix    = sparse.hstack([text_weighted, attr_weighted]).tocsr()

n_text_dims = text_matrix.shape[1]
n_attr_dims = attr_matrix.shape[1]
print(f"poi matrix: {poi_matrix.shape}  ({n_text_dims} text + {n_attr_dims} attr, β={TEXT_WEIGHT})")

sparse.save_npz(os.path.join(MODELS_DIR, "poi_matrix.npz"), poi_matrix)
joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))

index_df = pd.DataFrame({
    "row_index":       range(len(clean)),
    "google_place_id": clean["google_place_id"],
    "name":            clean["name"],
    "category":        clean["category"],
    "location_zone":   feats["location_zone"] if "location_zone" in feats.columns else "",
    "weighted_score":  feats["weighted_score"],
    "average_rating":  clean["average_rating"],
    "review_count":    clean["review_count"],
})
index_df.to_csv(os.path.join(MODELS_DIR, "poi_index.csv"), index=False)

metadata = {
    "n_pois":      int(len(clean)),
    "vocab_size":  int(len(vectorizer.vocabulary_)),
    "n_text_dims": int(n_text_dims),
    "n_attr_dims": int(n_attr_dims),
    "total_dims":  int(poi_matrix.shape[1]),
    "params": {
        "MAX_FEATURES": MAX_FEATURES,
        "NGRAM_RANGE":  list(NGRAM_RANGE),
        "MIN_DF":       MIN_DF,
        "STOP_WORDS":   STOP_WORDS,
        "TEXT_WEIGHT":  TEXT_WEIGHT,
    },
    "attribute_columns": attr_cols,
    "ranking_strategy":  "two-stage: cosine similarity → re-rank with weighted_score",
}
with open(os.path.join(MODELS_DIR, "vector_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("saved poi_matrix.npz, tfidf_vectorizer.pkl, poi_index.csv, vector_metadata.json")
