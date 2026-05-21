import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CF_DIR     = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(os.path.dirname(CF_DIR), "data")
MODELS_DIR = os.path.join(CF_DIR, "models")

_R_HAT    = np.load(os.path.join(MODELS_DIR, "R_hat.npy"))
_R        = np.load(os.path.join(MODELS_DIR, "R.npy"))
_USER_IDX = pd.read_csv(os.path.join(MODELS_DIR, "user_index.csv")).set_index("user_id")["row_index"].to_dict()
_POI_IDX  = pd.read_csv(os.path.join(MODELS_DIR, "poi_index.csv"))
_POI_META = pd.read_csv(os.path.join(DATA_DIR, "clean_merged_cebu_pois_v2.csv")).set_index("google_place_id")


def recommend(user_id, top_n=10, exclude_rated=True):
    """Return top-N POI recommendations for a synthetic user."""
    if user_id not in _USER_IDX:
        raise ValueError(f"unknown user: {user_id}")

    u      = _USER_IDX[user_id]
    scores = _R_HAT[u].copy()

    if exclude_rated:
        scores[_R[u] > 0] = -np.inf

    top_idx = np.argsort(scores)[::-1][:top_n]

    rows = []
    for rank, idx in enumerate(top_idx, 1):
        gpid = _POI_IDX.iloc[idx]["google_place_id"]
        meta = _POI_META.loc[gpid] if gpid in _POI_META.index else {}
        rows.append({
            "rank":             rank,
            "name":             meta.get("name", ""),
            "category":         meta.get("category", ""),
            "predicted_rating": round(float(scores[idx]), 3),
            "google_place_id":  gpid,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    sample = list(_USER_IDX.keys())[0]
    print(f"recommendations for {sample}:")
    print(recommend(sample, top_n=5).to_string(index=False))
