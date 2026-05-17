# Record Matcher

Match business records between two CSV datasets by name and address. Output: one CSV with `id_1, id_2` pairs.

The two inputs have different shapes:
- `dataset_1` (combined-address style): `id, name, address` where `address` is one free-form string
- `dataset_2` (split-fields style): `id, account_name, owner_name, name, street, city, zip`

## Quick Start

```bash
pip install -r requirements.txt
python3 match_records.py --config config.yml
```

Python 3.10+ required. Single entrypoint, single config file, everything else is library code.

## Inputs and Outputs

| File | Direction | Purpose |
|---|---|---|
| `dataset_1.csv` | in | The combined-address dataset |
| `dataset_2.csv` | in | The split-fields dataset |
| `config.yml` | in | All settings (paths, weights, methods, blocking) |
| `matches.csv` | out | Final pairs: `id_1, id_2` |
| `matches_debug.csv` | out | Pairs with per-field scores for diagnostics |
| `rejected_dataset_*.csv` | out | Rows that failed ingestion validation |
| `unmatched_dataset_*.csv` | out | Rows that passed ingestion but found no partner |

The two unmatched files close the accounting loop: `matched + unmatched = post-ingestion total` on each side, always.

## Architecture

```
match_records.py    Orchestrator + Config + normalization + scoring + stage helpers
config.yml          Single source of truth — all knobs live here
ingestion.py        DataFrame-first validation + rejection logging
blocking.py         iter_chunks(left, right, key) — yields filtered chunks
assignment.py       6 pluggable methods + cardinality registry + min-id anchor
requirements.txt    pandas, numpy, pyyaml, rapidfuzz, scipy
```

## Pipeline (7 stages, top of `run()`)

1. **Read** — `pd.read_csv` for both datasets (inline; swap to `read_parquet` is a 2-line change)
2. **Ingest** — required-field + uniqueness validation; rejected rows kept for audit
3. **Write rejection logs** — `rejected_dataset_*.csv`
4. **Normalize** — `prep_combined` / `prep_split` produce comparable columns + blocking keys
5. **Match** — blocking (optional) + scoring + assignment + min-id anchor (optional)
6. **Write outputs** — `matches.csv` and `matches_debug.csv`
7. **Write unmatched** — `unmatched_dataset_*.csv`; log proves the totals add up

## Normalization

Per `Normalizer` class in `match_records.py`, fully config-driven:

- `name()`: uppercase, strip punctuation, drop business suffixes (`INC`, `LLC`, …) so `"Joe's Pizza LLC"` and `"Joe's Pizza Inc"` look the same
- `text()`: uppercase, collapse whitespace, expand street abbreviations (`ST → STREET`, `AVE → AVENUE`, …)
- `zip5()`: extract trailing 5-digit zip from any format (`"90062"`, `"900621234"`, `"90062-1234"`)

Both `prep_combined` and `prep_split` produce a consistent set of normalized columns plus blocking keys:

| Column | combined | split | Purpose |
|---|---|---|---|
| `name_norm` | from `name` | — | Per-strategy comparison field |
| `name1_norm`, `name2_norm`, `name3_norm` | — | from `account_name`, `owner_name`, `name` | dataset_2 has 3 name slots |
| `street_norm`, `city_norm` | parsed from `address` | from raw fields | Per-field scoring |
| `zip5` | parsed | extracted | Used in scoring + as blocking key |
| `zip3` | first 3 of zip5 | first 3 of zip5 | Blocking key — absorbs same-region typos |
| `address_norm` | from raw | constructed: street + city + zip5 | For `concatenated` strategy |
| `city_zip` | `city + "|" + zip5` | same | Composite blocking key |
| `city_zip3` | `city + "|" + zip3` | same | **Default blocking key** |

## Blocking

Blocking groups records by a column value so we only score pairs that share it. Massive reduction in candidates — on the test data, 93–97% fewer pairs to score.

**Default key: `city_zip3`** — city plus the first 3 digits of zip. Tighter than blocking on either field alone, but loose enough to absorb data-entry errors within a small region. Catches the real case where the same business had zip `93401` in one dataset and `93405` in the other — both have zip3 `934`, so they block together.

Auto-enables when `max(len(split), len(combined)) > enable_above_rows` (default 1000). For larger datasets it kicks in automatically; for small test data it's skipped.

### How `iter_chunks` works

```python
for value in shared_values_on_both_sides:
    yield value, left[left[key] == value], right[right[key] == value]
```

Chunks preserve their original pandas `.index`, which is the local-to-global index translation for free.

### How `_run_blocked` uses it

```python
for key in config.blocking_keys:
    for _value, split_chunk, combined_chunk in blocking.iter_chunks(split, combined, key):
        # SAME score_and_assign call _run_full uses, on a chunk
        chunk_pairs, _, chunk_field_scores = score_and_assign(
            split_chunk.reset_index(drop=True),
            combined_chunk.reset_index(drop=True),
            config,
        )
        # Translate local positional indices back to global via chunk.index
        # Merge into pair_scores keeping max score per pair
```

Multi-key blocking iterates each key separately; a pair scored under multiple keys keeps its best score.

### Blocking strategy comparison

| Key | Candidates | Recall | Notes |
|---|---|---|---|
| `zip5` | 40 | 38/39 | Misses cross-zip typos |
| `zip3` | 77 | 39/39 | Catches same-region typos, looser |
| `city_norm` | 57 | 39/39 | Captures city-level matches |
| `city_zip` (AND on full zip) | 40 | 38/39 | Strict — fails on zip typos |
| `city_zip3` ✓ | **57** | **39/39** | Default — strict but zip3 absorbs typos |
| `[zip5, city_norm]` (OR) | 97 | 39/39 | More candidates, same recall |

## Scoring

Two strategies, switchable via `scoring.strategy`:

**`per_field`** (default) — score name, street, city, zip independently, then weighted average. Per-field diagnostics in `matches_debug.csv`. Default weights: `name 0.40, street 0.40, city 0.10, zip 0.10`.

**`concatenated`** — score full address strings end-to-end. Default weights: `name 0.45, address 0.45, zip 0.10`. More robust when address parsing might fail or for free-form addresses.

NaN-aware weighted average: missing fields don't penalize, weight redistributes across present fields.

### Similarity functions (rapidfuzz-backed)

| Function | When to use |
|---|---|
| `token_set_ratio` | Default for names — tolerates extra tokens (`"LLC"`, `"Italian Restaurant"`) |
| `token_sort_ratio` | Default for addresses — every token counts, order doesn't |
| `ratio` | Plain edit-distance ratio |
| `partial_ratio` | Substring matching |
| `jaccard` / `jaccard_ngram` | Set-based similarity |
| `exact_match` | Binary 100/0 — used for zip5 |

### `field_mappings`

Per-strategy declaration of which normalized column(s) feed which scoring field. dataset_2's three name columns are MAX-pooled against dataset_1's one — whichever of the three best matches dataset_1's name wins.

## Assignment Methods

After scoring produces an NxM matrix, an assignment method picks which pairs are "matches". Six methods, three cardinality classes:

| Method | Cardinality | When to use |
|---|---|---|
| `hungarian` | 1:1 | Globally optimal total score; deterministic |
| `greedy_mutual` (default) | 1:1 | Pick highest pair, lock row+col, repeat. Same as Hungarian when contrast is high; faster |
| `mutual_best` | 1:1 | High precision: pairs that are mutually argmax; drops ambiguous cases |
| `one_per_row` | 1:many | Each row's argmax wins; cols can repeat. Branch → HQ |
| `one_per_col` | many:1 | Each col's argmax wins; rows can repeat. HQ → branch |
| `threshold_only` | many:many | Every pair above threshold |

### Min-ID anchor (optional)

Configured via `assignment.anchor_by_min_id_on: combined | split | null`. Collapses ambiguity by **smallest id wins**, not by score:

- After the chosen method produces pairs, group by the non-parent side
- For each group, keep only the pair where the partner on the parent side has the smallest id
- Deterministic across runs; ignores float precision; matches "lower id = canonical" domain rules

Typical use: `threshold_only + anchor_by_min_id_on: combined`. With 1:1 methods it's a no-op (no ambiguity to break).

## Configuration Reference

```yaml
paths:                    # CSV in/out
logging.level:            # DEBUG | INFO | WARNING
scoring.strategy:         # per_field | concatenated
scoring.match_threshold:  # default 75
scoring.per_field:
  weights:                # per-field weights (sum to 1)
  similarity:              # per-field similarity function
  field_mappings:         # which columns feed which field
scoring.concatenated:     # mirror of per_field but with combined address
assignment:
  method:                 # 6 options (see table above)
  anchor_by_min_id_on:    # combined | split | null
blocking:
  enable_above_rows:      # auto-threshold; 0 = always block
  keys:                   # column(s) to block on
ingestion:
  mode:                   # iterative | vectorized
  required_fields:        # by source
  unique_fields:          # by source
normalization:
  business_suffixes:      # for name normalization
  street_abbreviations:   # ST -> STREET etc
```

## Design Decisions (the "why")

### Why DataFrame-first ingestion
Ingestion takes a DataFrame, not a file path. Reading is the orchestrator's concern; validation is ingestion's. Switching from CSV to Parquet is a 2-line change in `run()` — ingestion doesn't need to know.

### Why no within-dataset dedup
A business with three name slots (account_name, owner_name, name) has those slots intentionally. Capturing those variants is the matching signal, not noise. Dedup would collapse the signal we depend on.

### Why blocking is intentionally minimal
Blocking is just "filter the DataFrame by a column value, score the chunk, merge results." The `iter_chunks` generator yields tuples; there's no `Block` dataclass, no closure indirection. The local-to-global index translation falls out for free because pandas preserves the original index on a boolean filter.

### Why min-id anchor is a post-processing step, not a method
Cardinality (how many partners per side) is orthogonal to the tiebreaker rule (which partner wins ties). 1:1 methods break ties by score. Anchor breaks ties by id. Layering them lets you say "many:many matching, then deterministic collapse by id" — `threshold_only + anchor_by_min_id_on=combined`.

### Why `field_score_getter` is a lambda in `_match`
The two execution paths produce different data shapes:
- `_run_full` returns `{field: NxM_matrix}` — dict of full matrices
- `_run_blocked` returns `{(i,j): {field: score}}` — dict keyed by pair

The lambda hides this distinction behind a uniform interface `(i, j) → {field: score}` so the downstream debug writer doesn't need to know which branch ran.

### Why `city_zip3` is the default blocking key
On test data: same 39/39 recall as the OR strategy with 41% fewer candidates. Catches Olive Garden (zip 93401 vs 93405 in source data — both have zip3 `934`). Tighter than zip-only or city-only blocking, but the zip3 prefix is loose enough to absorb intra-region zip typos.

### Why "drop is not delete"
Every record that doesn't make it into `matches.csv` is preserved:
- Failed ingestion → `rejected_dataset_*.csv` with reason
- Passed ingestion but no partner → `unmatched_dataset_*.csv`

An auditor can verify nothing was silently lost.

## Scaling

At 39 rows everything is fast. At 39 million rows:

| Stage | What changes |
|---|---|
| Read | Switch to Parquet (2-line change in `run()`) |
| Ingestion | Switch `mode: iterative` → `mode: vectorized` (~2x faster) |
| Blocking | The same logic — but the chunk count grows, each chunk stays small |
| Scoring | Bottleneck shifts to per-chunk scoring; trivially parallelizable across chunks |
| Assignment | Hungarian is O(n^3), unusable on full matrix; greedy_mutual on the chunk is fine |

The orchestration can switch to Spark/Dask, but the per-chunk logic in `score_and_assign` doesn't change — the same DataFrame goes in, the same pair list comes out. Blocking is the load-bearing piece.

## Common Operations

**Switch input from CSV to Parquet:**
1. Change `pd.read_csv` → `pd.read_parquet` (2 lines in `run()`)
2. Update paths in `config.yml`
3. Add `pyarrow` to `requirements.txt`

**Add a new similarity function:**
1. Add a `def my_sim(a, b): ...` somewhere in `match_records.py`
2. Register it in `SIMILARITY_FUNCTIONS`
3. Reference by name in `config.yml` under `scoring.<strategy>.similarity.<field>`

**Add a new assignment method:**
1. Add `def my_method(scores, threshold): ...` to `assignment.py`
2. Register in `METHODS` dict and `METHOD_CARDINALITY` dict
3. Reference by name in `config.yml` under `assignment.method`

**Change blocking strategy:**
- Just edit `blocking.keys` in `config.yml` — any column added in `prep_combined`/`prep_split` is usable

## Interview Talking Points

**On the architecture:** "I drew the I/O boundary at the top of `run()` — ingestion operates on DataFrames, not paths. That made the pipeline format-agnostic. Switching CSV to Parquet is two lines. All the matching logic is on DataFrames, so it doesn't care where they came from."

**On blocking:** "Blocking is a filter loop. For each blocking key, I find values present on both sides, filter both DataFrames to that value, and run the same `score_and_assign` call I'd run on the full dataset — just on a chunk. The default key is `city_zip3` — city plus the first 3 digits of zip. Tighter than blocking on either alone, but the zip3 prefix tolerates intra-region zip typos, which catches a real Olive Garden case in the test data where the same business has different zip codes in the two sources."

**On assignment methods:** "I support six methods across four cardinality classes — 1:1, 1:many, many:1, many:many. Choosing the right one is a domain question, not a code one: are these snapshots of the same canonical entity list (1:1), or branches mapping to a parent HQ (1:many)? Hungarian gives globally optimal 1:1; greedy_mutual is faster and gives the same answer when contrast is high. Mutual_best is conservative — drops ambiguous pairs. Threshold_only is many:many for downstream clustering."

**On the min-id anchor:** "Cardinality and tiebreaker are orthogonal. The 1:1 methods break ties by score; the min-id anchor breaks ties by id, ignoring score. Layering them lets you say 'threshold matching, then collapse to a deterministic parent by id' — which is what auditors want because the rule is explicit and inspectable."

**On scaling:** "The bottleneck shifts to scoring as N grows, but scoring is per-chunk and embarrassingly parallel. Hungarian is O(n^3) — fine on a chunk of 50 records, dead on a chunk of 50,000. Greedy_mutual is the right choice at scale. Blocking is the load-bearing piece; getting the keys right is more important than the assignment method."

**On what I'd ask the interviewer:**
1. What's the cardinality assumption — same canonical entity list, or hierarchical?
2. What's the cost of a false positive vs a false negative?
3. What's the expected scale?
4. Is this a one-shot reconciliation or an ongoing pipeline?
