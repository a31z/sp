import os
import json
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from build_user_profile import build_user_vector, VALID_CATEGORIES, VALID_ZONES

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CBF_DIR    = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(CBF_DIR, "models")

_POI_MATRIX = sparse.load_npz(os.path.join(MODELS_DIR, "poi_matrix.npz"))
_INDEX      = pd.read_csv(os.path.join(MODELS_DIR, "poi_index.csv"))
with open(os.path.join(MODELS_DIR, "vector_metadata.json")) as f:
    _META = json.load(f)


def recommend(
    categories=None,
    query=None,
    zones=None,
    alpha=0.3,
    top_n=10,
    top_k=30,
    exclude_ids=None,
    verbose=False,
):
    """
    Return top-N POI recommendations.

    alpha       — stage-2 blend (0=pure relevance, 1=pure popularity, default 0.3)
    top_k       — candidate pool for stage-1; must be >= top_n
    exclude_ids — google_place_ids to skip (e.g. already-visited POIs)
    """
    if top_k < top_n:
        top_k = top_n

    user_vec = build_user_vector(categories=categories, query=query, zones=zones, verbose=verbose)
    sims = cosine_similarity(user_vec, _POI_MATRIX).flatten()

    if exclude_ids:
        sims[_INDEX["google_place_id"].isin(set(exclude_ids)).values] = -np.inf

    candidate_idx = np.argsort(sims)[::-1][:top_k]

    cand = _INDEX.iloc[candidate_idx].copy()
    cand["similarity"]  = sims[candidate_idx]
    cand["final_score"] = (1 - alpha) * cand["similarity"] + alpha * cand["weighted_score"]
    cand = cand.sort_values("final_score", ascending=False).head(top_n).reset_index(drop=True)
    cand.insert(0, "rank", cand.index + 1)

    return cand[[
        "rank", "name", "category", "location_zone",
        "similarity", "weighted_score", "final_score",
        "average_rating", "review_count", "google_place_id",
    ]]


if __name__ == "__main__":
    print(f"matrix: {_POI_MATRIX.shape}  index: {len(_INDEX)} POIs")

    print("\nbeach + nature, alpha=0.3")
    print(recommend(categories=["beach", "nature"], alpha=0.3, top_n=5).to_string(index=False))

    print("\nhistory + text query, alpha=0.3")
    print(recommend(
        categories=["history", "museum"],
        query="spanish colonial architecture old churches",
        alpha=0.3, top_n=5,
    ).to_string(index=False))

    print("\nsame query, alpha=0.7 (favour popular)")
    print(recommend(
        categories=["history", "museum"],
        query="spanish colonial architecture old churches",
        alpha=0.7, top_n=5,
    ).to_string(index=False))
