import os
import re
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(os.path.dirname(SCRIPT_DIR), "data")


def load_data():
    clean  = pd.read_csv(os.path.join(DATA_DIR, "clean_merged_cebu_pois.csv"))
    merged = pd.read_csv(os.path.join(DATA_DIR, "merged_cebu_pois.csv"))

    merged = (merged[merged["google_place_id"].isin(clean["google_place_id"])]
              .copy()
              .set_index("google_place_id"))

    print(f"loaded {len(clean)} clean rows, {len(merged)} merged rows aligned")
    return clean, merged


def clean_text(t):
    if pd.isna(t):
        return ""
    return re.sub(r"\s+", " ", str(t)).strip()


def humanise_types(types_str):
    NOISE = {"point_of_interest", "establishment"}
    if pd.isna(types_str):
        return ""
    parts = [
        p.strip().replace("_", " ")
        for p in str(types_str).split(",")
        if p.strip() and p.strip() not in NOISE
    ]
    return " ".join(parts)


def dedupe_segments(segments):
    seen, out = set(), []
    for seg in segments:
        if not seg:
            continue
        key = re.sub(r"[^a-z0-9]+", "", seg.lower())
        if key in seen or any(key in s for s in seen):
            continue
        seen.add(key)
        out.append(seg)
    return out


def enrich_description(row, merged):
    gpid = row["google_place_id"]
    bits = [
        clean_text(row["name"]),
        clean_text(row["category"]),
        clean_text(row["description"]),
    ]

    if gpid in merged.index:
        m = merged.loc[gpid]
        bits.append(humanise_types(m.get("types")))
        for col in ("sugbo_description", "sugbo_what_to_expect", "ta_description"):
            val = m.get(col)
            if pd.notna(val) and str(val).strip():
                bits.append(clean_text(val))

    return re.sub(r"\s+", " ", " ".join(dedupe_segments(bits))).strip()


def print_stats(original, enriched):
    orig_wc     = original.str.split().str.len()
    enriched_wc = enriched.str.split().str.len()
    print(f"word counts        min  median  mean  max")
    print(f"  original         {orig_wc.min():>3}  {orig_wc.median():>6.0f}  {orig_wc.mean():>4.0f}  {orig_wc.max():>4}")
    print(f"  enriched         {enriched_wc.min():>3}  {enriched_wc.median():>6.0f}  {enriched_wc.mean():>4.0f}  {enriched_wc.max():>4}")
    print(f"  avg gain: {enriched_wc.mean() - orig_wc.mean():.1f} words / POI")


def main():
    clean, merged = load_data()

    clean["description_original"] = clean["description"]
    clean["description"] = clean.apply(
        lambda r: enrich_description(r, merged), axis=1
    )

    print_stats(clean["description_original"], clean["description"])

    out_cols = [
        "name", "google_place_id", "search_query",
        "latitude", "longitude", "address",
        "average_rating", "review_count",
        "category", "description", "description_original",
    ]
    out_path = os.path.join(DATA_DIR, "clean_merged_cebu_pois_v2.csv")
    clean[out_cols].to_csv(out_path, index=False)
    print(f"\nsaved {len(clean)} rows → {out_path}")

    print("\nsamples:")
    for i in range(2):
        r = clean.iloc[i]
        print(f"\n  {r['name']}")
        print(f"    before: {r['description_original']}")
        print(f"    after:  {r['description']}")


if __name__ == "__main__":
    main()
