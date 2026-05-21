import os
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EVAL_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
CBF_DIR  = os.path.join(ROOT_DIR, "content-filtering")
CF_DIR   = os.path.join(ROOT_DIR, "collaborative-filtering")

sys.path.insert(0, os.path.join(CBF_DIR, "scripts"))
from build_user_profile import build_user_vector

K_VALUES       = [5, 10]
ALPHA          = 0.5   # hybrid blend: α*CBF + (1-α)*CF
LIKE_THRESHOLD = 3.5   # minimum rating to count as liked in held-out set

# CF artifacts
R         = np.load(os.path.join(CF_DIR, "models", "R.npy"))
R_hat     = np.load(os.path.join(CF_DIR, "models", "R_hat.npy"))
test_mask = np.load(os.path.join(CF_DIR, "models", "test_mask.npy"))

cf_user_idx = (pd.read_csv(os.path.join(CF_DIR, "models", "user_index.csv"))
               .set_index("user_id")["row_index"].to_dict())
cf_poi_ids  = pd.read_csv(os.path.join(CF_DIR, "models", "poi_index.csv"))["google_place_id"].tolist()

# CBF artifacts
cbf_matrix  = sparse.load_npz(os.path.join(CBF_DIR, "models", "poi_matrix.npz"))
cbf_poi_ids = pd.read_csv(os.path.join(CBF_DIR, "models", "poi_index.csv"))["google_place_id"].tolist()

# user type lookup
ratings    = pd.read_csv(os.path.join(DATA_DIR, "synthetic_ratings.csv"))
user_types = ratings.drop_duplicates("user_id").set_index("user_id")["user_type"].to_dict()

TYPE_CATEGORIES = {
    "beach_lover":   ["beach", "nature"],
    "history_buff":  ["history", "museum"],
    "foodie":        ["food_drink"],
    "thrill_seeker": ["entertainment", "nature"],
    "spiritual":     ["religious", "history"],
    "explorer":      ["attraction", "nature", "museum", "beach"],
}

# training-rated items per user (held-out zeroed out, so these are safe to exclude)
R_train = R.copy()
R_train[test_mask] = 0


def precision_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & relevant) / k

def recall_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0

def ndcg_at_k(ranked, relevant, k):
    dcg   = sum(1.0 / np.log2(i + 2) for i, item in enumerate(ranked[:k]) if item in relevant)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0

def top_k_by_score(scores_dict, exclude, k):
    return [
        gpid for gpid, _ in sorted(scores_dict.items(), key=lambda x: -x[1])
        if gpid not in exclude
    ][:k]


results = []

for user_id, u_idx in cf_user_idx.items():
    user_type  = user_types.get(user_id, "explorer")
    categories = TYPE_CATEGORIES.get(user_type, [])

    relevant = {
        cf_poi_ids[j]
        for j in np.where(test_mask[u_idx])[0]
        if R[u_idx, j] >= LIKE_THRESHOLD
    }
    if not relevant:
        continue

    train_gpids = {cf_poi_ids[j] for j in np.where(R_train[u_idx] > 0)[0]}

    # CF scores
    cf_scores = {cf_poi_ids[j]: float(R_hat[u_idx, j]) for j in range(len(cf_poi_ids))}

    # CBF scores
    user_vec   = build_user_vector(categories=categories)
    sims       = cosine_similarity(user_vec, cbf_matrix).flatten()
    cbf_scores = {cbf_poi_ids[j]: float(sims[j]) for j in range(len(cbf_poi_ids))}

    # normalise to [0, 1] for hybrid blend
    def norm(d):
        vals = np.array(list(d.values()))
        lo, hi = vals.min(), vals.max()
        normed = (vals - lo) / (hi - lo + 1e-9)
        return dict(zip(d.keys(), normed))

    cf_norm  = norm(cf_scores)
    cbf_norm = norm(cbf_scores)

    common = set(cf_norm) & set(cbf_norm)
    hybrid_scores = {g: ALPHA * cbf_norm[g] + (1 - ALPHA) * cf_norm[g] for g in common}

    k_max = max(K_VALUES)
    cf_ranked     = top_k_by_score(cf_scores,     train_gpids, k_max)
    cbf_ranked    = top_k_by_score(cbf_scores,    train_gpids, k_max)
    hybrid_ranked = top_k_by_score(hybrid_scores, train_gpids, k_max)

    row = {"user_id": user_id, "user_type": user_type, "n_relevant": len(relevant)}
    for k in K_VALUES:
        for label, ranked in [("cbf", cbf_ranked), ("cf", cf_ranked), ("hybrid", hybrid_ranked)]:
            row[f"{label}_p@{k}"]    = precision_at_k(ranked, relevant, k)
            row[f"{label}_r@{k}"]    = recall_at_k(ranked,    relevant, k)
            row[f"{label}_ndcg@{k}"] = ndcg_at_k(ranked,      relevant, k)

    results.append(row)

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(EVAL_DIR, "results.csv"), index=False)

# summary table
mean = results_df[[c for c in results_df.columns if "@" in c]].mean()

print(f"\n{'model':<8}  {'metric':<10}  {'@5':>7}  {'@10':>7}")
print("-" * 38)
for model in ["cbf", "cf", "hybrid"]:
    for metric, label in [("p", "precision"), ("r", "recall"), ("ndcg", "ndcg")]:
        v5  = mean.get(f"{model}_{metric}@5",  0)
        v10 = mean.get(f"{model}_{metric}@10", 0)
        print(f"{model:<8}  {label:<10}  {v5:>7.4f}  {v10:>7.4f}")
    print()

# summary csv
summary_rows = []
for model in ["cbf", "cf", "hybrid"]:
    for metric, label in [("p", "precision"), ("r", "recall"), ("ndcg", "ndcg")]:
        for k in K_VALUES:
            summary_rows.append({
                "model":  model,
                "metric": label,
                "k":      k,
                "value":  round(mean[f"{model}_{metric}@{k}"], 4),
            })
pd.DataFrame(summary_rows).to_csv(os.path.join(EVAL_DIR, "summary.csv"), index=False)

print(f"evaluated {len(results_df)} users  →  evaluation/results.csv, evaluation/summary.csv")
