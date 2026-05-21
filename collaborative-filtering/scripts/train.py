import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CF_DIR     = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(os.path.dirname(CF_DIR), "data")
MODELS_DIR = os.path.join(CF_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

N_FACTORS    = 20
TEST_FRAC    = 0.2
RANDOM_STATE = 42

ratings = pd.read_csv(os.path.join(DATA_DIR, "synthetic_ratings.csv"))
pois    = pd.read_csv(os.path.join(DATA_DIR, "poi_features.csv"))
print(f"loaded {len(ratings)} ratings from {ratings['user_id'].nunique()} users")

users   = sorted(ratings["user_id"].unique())
poi_ids = pois["google_place_id"].tolist()

user_idx = {u: i for i, u in enumerate(users)}
poi_idx  = {p: i for i, p in enumerate(poi_ids)}

R = np.zeros((len(users), len(poi_ids)))
for _, row in ratings.iterrows():
    u = user_idx.get(row["user_id"])
    p = poi_idx.get(row["google_place_id"])
    if u is not None and p is not None:
        R[u, p] = row["rating"]

# hold out TEST_FRAC of each user's ratings for evaluation
rng     = np.random.default_rng(RANDOM_STATE)
R_train = R.copy()
test_mask = np.zeros_like(R, dtype=bool)

for u in range(len(users)):
    rated   = np.where(R[u] > 0)[0]
    n_test  = max(1, int(len(rated) * TEST_FRAC))
    held    = rng.choice(rated, size=n_test, replace=False)
    R_train[u, held] = 0
    test_mask[u, held] = True

# mean-center rows before SVD
row_means = np.array([
    R_train[u][R_train[u] > 0].mean() if (R_train[u] > 0).any() else 0
    for u in range(len(users))
])
R_centered = R_train.copy()
for u in range(len(users)):
    R_centered[u, R_centered[u] > 0] -= row_means[u]

svd = TruncatedSVD(n_components=N_FACTORS, random_state=RANDOM_STATE)
U   = svd.fit_transform(R_centered)   # (n_users, k)
Vt  = svd.components_                 # (k, n_pois)

R_hat = np.clip(U @ Vt + row_means[:, None], 1.0, 5.0)

np.save(os.path.join(MODELS_DIR, "R.npy"),        R)
np.save(os.path.join(MODELS_DIR, "R_hat.npy"),    R_hat)
np.save(os.path.join(MODELS_DIR, "test_mask.npy"), test_mask)
joblib.dump(svd, os.path.join(MODELS_DIR, "svd_model.pkl"))

pd.DataFrame({"user_id": users, "row_index": range(len(users))}).to_csv(
    os.path.join(MODELS_DIR, "user_index.csv"), index=False
)
pd.DataFrame({"google_place_id": poi_ids, "col_index": range(len(poi_ids))}).to_csv(
    os.path.join(MODELS_DIR, "poi_index.csv"), index=False
)

with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
    json.dump({
        "n_users":   len(users),
        "n_pois":    len(poi_ids),
        "n_factors": N_FACTORS,
        "test_frac": TEST_FRAC,
        "n_ratings": int((R > 0).sum()),
        "sparsity":  round(float(1 - (R > 0).sum() / R.size), 4),
    }, f, indent=2)

print(f"trained  users={len(users)}  pois={len(poi_ids)}  factors={N_FACTORS}")
print(f"held out {int(test_mask.sum())} ratings for evaluation ({TEST_FRAC:.0%} per user)")
print(f"saved R.npy, R_hat.npy, test_mask.npy, svd_model.pkl, user_index.csv, poi_index.csv, metadata.json")
