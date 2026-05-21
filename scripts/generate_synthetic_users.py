import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(os.path.dirname(SCRIPT_DIR), "data")

np.random.seed(42)

USER_TYPES = [
    {"name": "beach_lover",   "preferred": ["beach", "nature"],                         "count": 20},
    {"name": "history_buff",  "preferred": ["history", "museum"],                       "count": 15},
    {"name": "foodie",        "preferred": ["food_drink"],                              "count": 20},
    {"name": "thrill_seeker", "preferred": ["entertainment", "nature"],                 "count": 15},
    {"name": "spiritual",     "preferred": ["religious", "history"],                    "count": 15},
    {"name": "explorer",      "preferred": ["attraction", "nature", "museum", "beach"], "count": 15},
]

pois = pd.read_csv(os.path.join(DATA_DIR, "poi_features.csv"))
print(f"loaded {len(pois)} POIs")

records = []
user_id = 0

for utype in USER_TYPES:
    preferred = set(utype["preferred"])

    for _ in range(utype["count"]):
        for _, poi in pois.iterrows():
            is_preferred = poi["category"] in preferred

            # preferred POIs rated 60% of the time, non-preferred 10%
            if np.random.random() > (0.6 if is_preferred else 0.1):
                continue

            base = np.random.uniform(3.5, 5.0) if is_preferred else np.random.uniform(1.0, 3.0)
            # higher-quality POIs get a small nudge
            rating = base + 0.5 * poi["weighted_score"] + np.random.normal(0, 0.3)
            rating = float(np.clip(rating, 1.0, 5.0))

            records.append({
                "user_id":         f"u{user_id:03d}",
                "user_type":       utype["name"],
                "google_place_id": poi["google_place_id"],
                "rating":          round(rating, 2),
            })

        user_id += 1

ratings = pd.DataFrame(records)
ratings.to_csv(os.path.join(DATA_DIR, "synthetic_ratings.csv"), index=False)

n_users   = ratings["user_id"].nunique()
n_ratings = len(ratings)
sparsity  = 1 - n_ratings / (n_users * len(pois))
print(f"{n_users} users  {n_ratings} ratings  sparsity={sparsity:.2%}")
print(f"saved synthetic_ratings.csv")
