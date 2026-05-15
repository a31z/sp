import os
import json
import joblib
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CBF_DIR    = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(CBF_DIR, "models")

_VECTORIZER = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
with open(os.path.join(MODELS_DIR, "vector_metadata.json")) as f:
    _METADATA = json.load(f)

ATTR_COLS   = _METADATA["attribute_columns"]
TEXT_WEIGHT = _METADATA["params"]["TEXT_WEIGHT"]
N_TEXT_DIMS = _METADATA["n_text_dims"]
N_ATTR_DIMS = _METADATA["n_attr_dims"]

VALID_CATEGORIES = [c.replace("cat_", "")  for c in ATTR_COLS if c.startswith("cat_")]
VALID_ZONES      = [z.replace("zone_", "") for z in ATTR_COLS if z.startswith("zone_")]


def build_user_vector(categories=None, query=None, zones=None, verbose=False):
    """Return a 1×N sparse vector aligned with the POI matrix."""
    if query and query.strip():
        text_vec = _VECTORIZER.transform([query])
        text_vec = normalize(text_vec, norm="l2", axis=1)
    else:
        text_vec = sparse.csr_matrix((1, N_TEXT_DIMS))

    attr_vec = np.zeros((1, N_ATTR_DIMS))

    if categories:
        for c in categories:
            col = f"cat_{c.strip().lower()}"
            if col in ATTR_COLS:
                attr_vec[0, ATTR_COLS.index(col)] = 1.0
            elif verbose:
                print(f"unknown category: '{c}' (valid: {VALID_CATEGORIES})")

    if zones:
        for z in zones:
            col = f"zone_{z.strip().lower()}"
            if col in ATTR_COLS:
                attr_vec[0, ATTR_COLS.index(col)] = 1.0
            elif verbose:
                print(f"unknown zone: '{z}' (valid: {VALID_ZONES})")

    # skip normalise on all-zero attr to avoid NaN
    if attr_vec.sum() > 0:
        attr_vec = normalize(attr_vec, norm="l2", axis=1)
    attr_vec = sparse.csr_matrix(attr_vec)

    return sparse.hstack([text_vec * TEXT_WEIGHT, attr_vec * (1 - TEXT_WEIGHT)]).tocsr()


if __name__ == "__main__":
    print(f"vocab: {N_TEXT_DIMS}  attrs: {N_ATTR_DIMS}  β={TEXT_WEIGHT}")
    print(f"categories: {VALID_CATEGORIES}")
    print(f"zones:      {VALID_ZONES}")

    cases = [
        {"label": "beach + nature",       "args": {"categories": ["beach", "nature"]}},
        {"label": "history + text query", "args": {"categories": ["history", "museum"], "query": "spanish colonial architecture and old churches"}},
        {"label": "cold start",           "args": {}},
        {"label": "zone-restricted",      "args": {"categories": ["nature"], "zones": ["south_cebu"]}},
        {"label": "unknown category",     "args": {"categories": ["beachs", "nature"], "verbose": True}},
    ]

    for case in cases:
        v = build_user_vector(**case["args"])
        norm_val = float(np.sqrt((v.multiply(v)).sum()))
        print(f"\n  {case['label']}: shape={v.shape}  nnz={v.nnz}  norm={norm_val:.3f}")
