## Requirements

```
pip install pandas numpy rapidfuzz
```

Python 3.9+.

---

## Scripts

### 1. `merge_cebu_pois.py`

Merges all four sources and produces the clean dataset.

```bash
python merge_cebu_pois.py
```

---

### 2. `enrich_descriptions.py`

Enriches the clean dataset's `description` field by combining text from Google + Sugbo + TA + humanised Google `types`, then dedupes overlapping segments.

```bash
python enrich_descriptions.py
```

Adds one column (`description_original`) so the v1 text is preserved for comparison.

---

## Order to run

```
1. merge_cebu_pois.py   →  produces clean + merged
2. enrich_descriptions.py    →  enriches clean into v2
```