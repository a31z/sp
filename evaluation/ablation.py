import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EVAL_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
CBF_DIR  = os.path.join(ROOT_DIR, "content-filtering")
CF_DIR   = os.path.join(ROOT_DIR, "collaborative-filtering")

# base data
clean   = pd.read_csv(os.path.join(DATA_DIR, "clean_merged_cebu_pois_v2.csv"))
feats   = pd.read_csv(os.path.join(DATA_DIR, "poi_features.csv"))
ratings = pd.read_csv(os.path.join(DATA_DIR, "synthetic_ratings.csv"))

clean = clean.set_index("google_place_id")
feats = feats.set_index("google_place_id")
common = clean.index.intersection(feats.index)
clean = clean.loc[common].reset_index()
feats = feats.loc[common].reset_index()

# CF artifacts
R         = np.load(os.path.join(CF_DIR, "models", "R.npy"))
test_mask = np.load(os.path.join(CF_DIR, "models", "test_mask.npy"))
cf_user_idx = (pd.read_csv(os.path.join(CF_DIR, "models", "user_index.csv"))
               .set_index("user_id")["row_index"].to_dict())
cf_poi_ids  = pd.read_csv(os.path.join(CF_DIR, "models", "poi_index.csv"))["google_place_id"].tolist()

R_train = R.copy()
R_train[test_mask] = 0

# CBF metadata
with open(os.path.join(CBF_DIR, "models", "vector_metadata.json")) as f:
    cbf_meta = json.load(f)
ATTR_COLS   = cbf_meta["attribute_columns"]
N_ATTR_DIMS = cbf_meta["n_attr_dims"]

user_types = ratings.drop_duplicates("user_id").set_index("user_id")["user_type"].to_dict()

TYPE_CATEGORIES = {
    "beach_lover":   ["beach", "nature"],
    "history_buff":  ["history", "museum"],
    "foodie":        ["food_drink"],
    "thrill_seeker": ["entertainment", "nature"],
    "spiritual":     ["religious", "history"],
    "explorer":      ["attraction", "nature", "museum", "beach"],
}

K_VALUES = [5, 10]

DEFAULTS = {
    "text_weight":    0.5,
    "max_features":   None,
    "ngram_range":    (1, 2),
    "min_df":         2,
    "n_factors":      20,
    "bayesian_m":     50,
    "alpha":          0.5,
    "like_threshold": 3.5,
}


def precision_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & relevant) / k

def recall_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0

def ndcg_at_k(ranked, relevant, k):
    dcg   = sum(1.0 / np.log2(i + 2) for i, item in enumerate(ranked[:k]) if item in relevant)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0

def top_k_by_score(scores_dict, exclude, k):
    return [g for g, _ in sorted(scores_dict.items(), key=lambda x: -x[1]) if g not in exclude][:k]

def norm_dict(d):
    vals = np.array(list(d.values()))
    lo, hi = vals.min(), vals.max()
    return dict(zip(d.keys(), (vals - lo) / (hi - lo + 1e-9)))


def build_user_vec(categories, n_text_dims, text_weight):
    attr_vec = np.zeros((1, N_ATTR_DIMS))
    for c in (categories or []):
        col = f"cat_{c.strip().lower()}"
        if col in ATTR_COLS:
            attr_vec[0, ATTR_COLS.index(col)] = 1.0
    if attr_vec.sum() > 0:
        attr_vec = normalize(attr_vec, norm="l2", axis=1)
    return sparse.hstack([
        sparse.csr_matrix((1, n_text_dims)) * text_weight,
        sparse.csr_matrix(attr_vec) * (1 - text_weight),
    ]).tocsr()


def build_cbf_matrix(text_weight, max_features, ngram_range, min_df):
    vec      = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range,
                               min_df=min_df, stop_words="english",
                               lowercase=True, strip_accents="unicode")
    text_mat = normalize(vec.fit_transform(clean["description"].fillna("")), norm="l2", axis=1)
    attr_mat = normalize(feats[ATTR_COLS].astype(float).values, norm="l2", axis=1)
    matrix   = sparse.hstack([text_mat * text_weight,
                               sparse.csr_matrix(attr_mat * (1 - text_weight))]).tocsr()
    return matrix, clean["google_place_id"].tolist(), text_mat.shape[1]


def train_cf(n_factors):
    row_means = np.array([
        R_train[u][R_train[u] > 0].mean() if (R_train[u] > 0).any() else 0
        for u in range(R_train.shape[0])
    ])
    R_c = R_train.copy()
    for u in range(R_train.shape[0]):
        R_c[u, R_c[u] > 0] -= row_means[u]
    svd = TruncatedSVD(n_components=n_factors, random_state=42)
    U   = svd.fit_transform(R_c)
    return np.clip(U @ svd.components_ + row_means[:, None], 1.0, 5.0)


def recompute_weighted_score(m):
    rating_norm = (clean["average_rating"] - 1) / 4.0
    C = rating_norm.mean()
    n = clean["review_count"]
    return ((n / (n + m)) * rating_norm + (m / (n + m)) * C).values


def run_eval(cbf_matrix, cbf_poi_ids, n_text_dims, text_weight, R_hat, alpha, like_threshold):
    results = []

    for user_id, u_idx in cf_user_idx.items():
        categories = TYPE_CATEGORIES.get(user_types.get(user_id, "explorer"), [])

        relevant = {
            cf_poi_ids[j]
            for j in np.where(test_mask[u_idx])[0]
            if R[u_idx, j] >= like_threshold
        }
        if not relevant:
            continue

        train_gpids = {cf_poi_ids[j] for j in np.where(R_train[u_idx] > 0)[0]}

        cf_scores  = {cf_poi_ids[j]: float(R_hat[u_idx, j]) for j in range(len(cf_poi_ids))}
        user_vec   = build_user_vec(categories, n_text_dims, text_weight)
        sims       = cosine_similarity(user_vec, cbf_matrix).flatten()
        cbf_scores = {cbf_poi_ids[j]: float(sims[j]) for j in range(len(cbf_poi_ids))}

        cf_n  = norm_dict(cf_scores)
        cbf_n = norm_dict(cbf_scores)
        hybrid_scores = {g: alpha * cbf_n[g] + (1 - alpha) * cf_n[g]
                         for g in set(cf_n) & set(cbf_n)}

        k_max = max(K_VALUES)
        row   = {}
        for label, scores in [("cbf", cbf_scores), ("cf", cf_scores), ("hybrid", hybrid_scores)]:
            ranked = top_k_by_score(scores, train_gpids, k_max)
            for k in K_VALUES:
                row[f"{label}_p@{k}"]    = precision_at_k(ranked, relevant, k)
                row[f"{label}_r@{k}"]    = recall_at_k(ranked,    relevant, k)
                row[f"{label}_ndcg@{k}"] = ndcg_at_k(ranked,      relevant, k)
        results.append(row)

    if not results:
        return {}
    df = pd.DataFrame(results)
    return {col: round(float(df[col].mean()), 4) for col in df.columns}


# pre-build defaults so they're not rebuilt on every iteration
print("building default artifacts …")
default_cbf_matrix, default_cbf_poi_ids, default_n_text = build_cbf_matrix(
    DEFAULTS["text_weight"], DEFAULTS["max_features"],
    DEFAULTS["ngram_range"], DEFAULTS["min_df"],
)
default_R_hat = np.load(os.path.join(CF_DIR, "models", "R_hat.npy"))

ABLATIONS = [
    # (param_name, values_to_try, which_artifacts_change)
    ("alpha",          [0.3, 0.5, 0.7],        "eval"),
    ("like_threshold", [3.0, 3.5, 4.0],        "eval"),
    ("text_weight",    [0.3, 0.5, 0.7],        "cbf"),
    ("max_features",   [500, 1000, None],       "cbf"),
    ("ngram_range",    [(1,1), (1,2), (1,3)],  "cbf"),
    ("min_df",         [1, 2, 3],              "cbf"),
    ("n_factors",      [10, 20, 50],           "cf"),
    ("bayesian_m",     [20, 50, 100],          "preproc"),
]

all_rows = []

for param_name, values, group in ABLATIONS:
    print(f"\n{param_name}:")
    for val in values:
        cfg = {**DEFAULTS, param_name: val}
        is_default = (val == DEFAULTS[param_name])

        if group == "eval":
            cbf_matrix, cbf_poi_ids, n_text = default_cbf_matrix, default_cbf_poi_ids, default_n_text
            R_hat = default_R_hat

        elif group == "cbf":
            cbf_matrix, cbf_poi_ids, n_text = build_cbf_matrix(
                cfg["text_weight"], cfg["max_features"],
                cfg["ngram_range"], cfg["min_df"],
            )
            R_hat = default_R_hat

        elif group == "cf":
            cbf_matrix, cbf_poi_ids, n_text = default_cbf_matrix, default_cbf_poi_ids, default_n_text
            R_hat = train_cf(cfg["n_factors"])

        elif group == "preproc":
            # bayesian_m changes weighted_score; affects CBF stage-2 in recommend.py
            # but not cosine-similarity-based eval — results will be constant across m values
            cbf_matrix, cbf_poi_ids, n_text = default_cbf_matrix, default_cbf_poi_ids, default_n_text
            R_hat = default_R_hat

        metrics = run_eval(
            cbf_matrix, cbf_poi_ids, n_text, cfg["text_weight"],
            R_hat, cfg["alpha"], cfg["like_threshold"],
        )

        row = {
            "param":      param_name,
            "value":      str(val),
            "is_default": is_default,
            **metrics,
        }
        all_rows.append(row)

        marker = " ← default" if is_default else ""
        print(f"  {str(val):>10}  hybrid_ndcg@10={metrics.get('hybrid_ndcg@10', 0):.4f}{marker}")

results_df = pd.DataFrame(all_rows)
results_df.to_csv(os.path.join(EVAL_DIR, "ablation_results.csv"), index=False)

# print summary table grouped by param
print("\n\n--- ablation summary (hybrid NDCG@10) ---\n")
metric = "hybrid_ndcg@10"
for param_name in results_df["param"].unique():
    sub = results_df[results_df["param"] == param_name][["value", "is_default", metric]]
    print(f"{param_name}")
    for _, r in sub.iterrows():
        marker = " *" if r["is_default"] else ""
        print(f"  {str(r['value']):>10}  {r[metric]:.4f}{marker}")
    print()

print(f"saved evaluation/ablation_results.csv  ({len(results_df)} configs)")
