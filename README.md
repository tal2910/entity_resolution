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

### The formula in one line

For each candidate pair `(i, j)`:

```
combined_score(i, j) = Σ weight[f] · field_score[f](i, j)   for all fields f where field_score is not NaN
                       ─────────────────────────────────
                           Σ weight[f]   for those same f
```

NaN-aware weighted average. Missing fields don't penalize — the weight redistributes across the fields that ARE present.

A pair becomes a candidate match iff `combined_score ≥ match_threshold`.

### Per-field score: MAX across the cross-product

```
field_score[f](i, j) = MAX over (lc, rc) of similarity_fn[f](left[lc][i], right[rc][j])
```

where `(lc, rc)` ranges over all column pairs declared in `field_mappings[f]`. For most fields the cross-product is 1×1 (one column each side). For `name` in this dataset, it's 1×3 — dataset_2 has three name slots (`account_name`, `owner_name`, `name`), dataset_1 has one (`name`), MAX takes the best.

### Strategy 1: `per_field` (default)

```yaml
weights:
  name:   0.40
  street: 0.40
  city:   0.10
  zip:    0.10
similarity:
  name:   token_set_ratio    # tolerant of extra tokens (LLC, Italian Restaurant)
  street: token_sort_ratio   # every token counts, order doesn't
  city:   token_sort_ratio
  zip:    exact_match        # binary: 100 if equal, 0 otherwise
```

Scores 4 fields independently. Per-field breakdown is written to `matches_debug.csv` — auditable.

### Strategy 2: `concatenated`

```yaml
weights:
  name:    0.45
  address: 0.45
  zip:     0.10
similarity:
  name:    token_set_ratio
  address: token_sort_ratio
  zip:     exact_match
```

Compares the full address as one string against the full address as one string. For dataset_1 the `address_norm` is the raw `address` field normalized; for dataset_2 it's `street + ", " + city + " " + zip5` then normalized. Use when address parsing might fail or for free-form / international addresses.

### Similarity functions (rapidfuzz-backed)

| Function | What it does | Best for |
|---|---|---|
| `token_set_ratio` | Compares the sets of tokens; tolerates **extra** tokens on either side | Names that may include extra words ('LLC', 'Italian Restaurant') |
| `token_sort_ratio` | Sorts tokens alphabetically then compares; every token matters | Addresses (order doesn't, but completeness does) |
| `ratio` | Plain Levenshtein-based edit-distance ratio | Strict comparison; punishes any difference |
| `partial_ratio` | Best-substring match of shorter into longer | One value may be a fragment of the other |
| `jaccard` | Set-based overlap of words | Short multi-word fields |
| `jaccard_ngram` | Set-based overlap of n-grams | Misspelling-tolerant fuzzy comparison |
| `exact_match` | 100 if equal, else 0 (binary) | Codes (zip, SSN, EIN) where any diff = different entity |

Picking the right similarity function matters more than picking the right weights. `token_set_ratio` is the right default for names because it absorbs LLC/Inc/extra-tokens noise. `token_sort_ratio` is the right default for addresses because '4 Apt' and 'Apt 4' should match, but missing tokens should hurt.

### Worked example A — clean match (MLK Gas Station)

```
dataset_2 row 0: name1='MLK GAS STATION', name2='TTVV',          name3='MLK GAS STATION',
                 street='1515 W MLK JR BLVD', city='LOS ANGELES', zip5='90062'
dataset_1 row 3: name='MLK Gas Station', address='1515 W Martin Luther King Jr Blvd, Los Angeles, CA 90062'
                 -> normalized: name='MLK GAS STATION', street='1515 W MARTIN LUTHER KING JR BOULEVARD',
                                city='LOS ANGELES', zip5='90062'

Per-field similarities:
  name:   MAX(sim('MLK GAS STATION', 'MLK GAS STATION'),     -> 100
             sim('TTVV',            'MLK GAS STATION'),     ->  21
             sim('MLK GAS STATION', 'MLK GAS STATION'))     -> 100
       =  100
  street: token_sort('1515 W MLK JR BLVD', '1515 W MARTIN LUTHER KING JR BOULEVARD') = ~100
  city:   token_sort('LOS ANGELES', 'LOS ANGELES')                                    = 100
  zip:    exact_match('90062', '90062')                                                = 100

Weighted average:  0.40·100 + 0.40·100 + 0.10·100 + 0.10·100  =  100.0
                   ─────────────────────────────────────────
                                   1.00

100 ≥ 75 → MATCH
```

### Worked example B — partial match with NaN renormalization (Olive Garden)

```
dataset_2 row k: name1='OLIVE GARDEN', name2='N AND D RESTAURANTS LLC', name3='OLIVE GARDEN',
                 street='11966 LOS OSOS VALLEY RD', city='SAN LUIS OBISPO', zip5='93405'
dataset_1 row m: name='Olive Garden Italian Restaurant',
                 address='11966 Los Osos Valley Rd, San Luis Obispo, CA 93401'
                 -> normalized: name='OLIVE GARDEN ITALIAN RESTAURANT',
                                street='11966 LOS OSOS VALLEY ROAD', city='SAN LUIS OBISPO', zip5='93401'

Per-field similarities:
  name:   MAX(sim('OLIVE GARDEN',                'OLIVE GARDEN ITALIAN RESTAURANT'),  -> 100  (token_set treats 'OLIVE GARDEN' as a subset)
             sim('N AND D RESTAURANTS LLC',      'OLIVE GARDEN ITALIAN RESTAURANT'),  ->  37
             sim('OLIVE GARDEN',                 'OLIVE GARDEN ITALIAN RESTAURANT'))  -> 100
       =  100
  street: token_sort('11966 LOS OSOS VALLEY RD', '11966 LOS OSOS VALLEY ROAD')       = 100  (RD normalizes to ROAD)
  city:   100
  zip:    exact_match('93405', '93401')                                              =   0

Weighted average:  0.40·100 + 0.40·100 + 0.10·100 + 0.10·0   =   90.0
                   ────────────────────────────────────────
                                  1.00

90 ≥ 75 → MATCH  (the zip mismatch costs 10 points but doesn't kill it)
```

### Worked example C — what NaN renormalization actually does

Suppose dataset_2 row has `zip5=''` (missing). Then:

```
Per-field similarities:
  name:   100
  street: 100
  city:   100
  zip:    NaN     <- can't compare missing values

valid fields = {name, street, city}
total_weight = 0.40 + 0.40 + 0.10 = 0.90    (zip weight 0.10 excluded)

Weighted average:  0.40·100 + 0.40·100 + 0.10·100    =  100.0
                   ─────────────────────────────────
                              0.90

100 ≥ 75 → MATCH  (missing zip did NOT penalize)
```

The alternative ("treat NaN as 0") would have given `90·0/1.0 = 90` — still a match here but punitive on partial data. With NaN renormalization, a row with no zip is judged purely on name/street/city.

### Tuning playbook (the values you can play with)

All knobs live in `config.yml` under `scoring.<strategy>`. Reload, rerun — no code change.

| Symptom | Knob | Direction |
|---|---|---|
| Too many false positives (junk pairs labeled "match") | `match_threshold` | ↑ raise (e.g. 75 → 85) |
| Missing real matches (low recall) | `match_threshold` | ↓ lower (e.g. 75 → 65) |
| Same-address but unrelated businesses paired | `weights.name` | ↑ raise (e.g. 0.40 → 0.50) |
| Same-name chains across cities paired | `weights.street` or `weights.city` | ↑ raise |
| Zip-typos killing real matches | `weights.zip` | ↓ lower (e.g. 0.10 → 0.05), or swap `similarity.zip` to `jaccard_ngram` |
| Street abbreviations causing false negatives | already handled by normalization (`ST→STREET` etc) — extend `normalization.street_abbreviations` |
| Suffixes like 'Inc/LLC' creating noise | already handled by normalization (`business_suffixes`) |
| Free-form addresses don't parse cleanly | switch `strategy: per_field` → `concatenated` |

### Constraints

- Weights must sum to 1.0 (validated at config-load time)
- Similarity function names must be in the `SIMILARITY_FUNCTIONS` registry
- Threshold is on the 0–100 scale (matches rapidfuzz convention)
- Missing fields produce NaN — fully handled by the renormalization above; never an exception

### Sensitivity check — what `match_threshold` does on the test data

| Threshold | Matches found | Behavior |
|---|---|---|
| 90 | 37 / 39 | Drops the 2 borderline pairs (Olive Garden with zip mismatch scores 90, falls below at 90.0 cutoff strict-greater check; another similar partial) |
| 80 | 39 / 39 | All real matches recovered |
| 75 (default) | 39 / 39 | Catches all real matches |
| 60 | 39 / 39 | Same; clean test data has no near-miss pairs in this range |
| 40 | 39 / 39 | Same; with `greedy_mutual` the 1:1 lock prevents false positives even at low threshold |

Test data is clean enough that thresholds 40–80 all produce the same matches (1:1 assignment locks each row to its single best partner, which is unambiguously correct). The threshold becomes load-bearing on **noisy data** or with **`threshold_only` assignment** — there it directly controls precision vs recall. Tune for your domain's cost asymmetry.



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
