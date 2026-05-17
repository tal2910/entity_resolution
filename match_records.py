"""
Match business records between two datasets by name + address.

Architecture
------------
* All tunables live in config.yml. The code never hard-codes weights,
  thresholds, abbreviations, suffixes, similarity functions, or strategy.

* Two scoring strategies are supported, picked by the top-level
  `strategy` field in the config:
      per_field    -> parse dataset_1's address into fields and score
                      name, street, city, and zip independently.
      concatenated -> score the name and the full address string on
                      each side, plus zip as a binary check.

* The similarity function is configurable per field. token_set_ratio,
  token_sort_ratio, plain ratio (Levenshtein), partial_ratio, token
  Jaccard, and character n-gram Jaccard are all available.

* Each scoring component is its own function returning an NxM matrix
  of 0..100 scores. NaN means "this field couldn't be compared on this
  pair"; the combiner is NaN-aware and redistributes that field's weight
  to the fields that did produce a comparison.

* Final assignment is pluggable via the `assignment.method` config key.
  Implementations live in assignment.py: hungarian, greedy_mutual,
  threshold_only, mutual_best.
"""

import re
import time
import logging
import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml
import pandas as pd
import numpy as np
from rapidfuzz import fuzz

import assignment
import blocking
import ingestion


logger = logging.getLogger(__name__)


# ===========================================================================
# Similarity functions
# ===========================================================================

def _jaccard_tokens(a: str, b: str) -> float:
    """Token-set Jaccard: |A intersect B| / |A union B|, scaled to 0..100."""
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb:
        return 0.0
    return 100.0 * len(ta & tb) / len(ta | tb)


def _jaccard_ngram(a: str, b: str, n: int = 3) -> float:
    """Character n-gram Jaccard (default n=3). Whitespace removed first."""
    def ngrams(s: str) -> set:
        s = s.replace(" ", "")
        if len(s) < n:
            return {s} if s else set()
        return {s[i:i + n] for i in range(len(s) - n + 1)}

    na, nb = ngrams(a), ngrams(b)
    if not na and not nb:
        return 0.0
    return 100.0 * len(na & nb) / len(na | nb)


def _exact_match(a: str, b: str) -> float:
    """Binary similarity: 100 if equal, 0 otherwise. Use for fields like
    zip5 where typo-tolerance is the wrong behavior."""
    return 100.0 if a == b else 0.0


SIMILARITY_FUNCTIONS = {
    "token_set_ratio":  fuzz.token_set_ratio,
    "token_sort_ratio": fuzz.token_sort_ratio,
    "ratio":            fuzz.ratio,
    "partial_ratio":    fuzz.partial_ratio,
    "jaccard":          _jaccard_tokens,
    "jaccard_ngram":    _jaccard_ngram,
    "exact_match":      _exact_match,
}


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass(frozen=True)
class Config:
    # Paths
    dataset_1_path: Path
    dataset_2_path: Path
    matches_path: Path
    matches_debug_path: Path | None
    rejected_dataset_1_path: Path | None
    rejected_dataset_2_path: Path | None
    unmatched_dataset_1_path: Path | None
    unmatched_dataset_2_path: Path | None

    # Logging
    log_level: str

    # Scoring
    strategy: str                       # "per_field" | "concatenated"
    weights_by_strategy: dict           # strategy -> {field: weight}
    similarity_by_strategy: dict        # strategy -> {field: similarity_name}
    field_mappings_by_strategy: dict    # strategy -> {field: {"combined": [...], "split": [...]}}
    match_threshold: float

    # Assignment
    assignment_method: str
    anchor_by_min_id_on: str | None    # "combined" | "split" | None

    # Blocking
    blocking_enable_above_rows: int
    blocking_keys: tuple

    # Ingestion
    required_fields_combined: tuple
    required_fields_split: tuple
    unique_fields_combined: tuple
    unique_fields_split: tuple
    ingestion_mode: str

    # Normalization
    business_suffixes: frozenset
    street_abbreviations: dict

    # Schema (drives prep_dataset)
    schema_combined: dict     # {parse_combined_address_from?, columns: {...}}
    schema_split: dict
    composites: dict          # {name: {parts: [...]}}

    @property
    def weights(self) -> dict:
        return self.weights_by_strategy[self.strategy]

    @property
    def similarity(self) -> dict:
        return self.similarity_by_strategy[self.strategy]

    @property
    def field_mappings(self) -> dict:
        return self.field_mappings_by_strategy[self.strategy]

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)
        s, n = raw["scoring"], raw["normalization"]

        # ----- Paths -----
        paths_cfg = raw.get("paths", {}) or {}
        try:
            dataset_1_path = Path(paths_cfg["dataset_1"])
            dataset_2_path = Path(paths_cfg["dataset_2"])
            matches_path   = Path(paths_cfg["matches"])
        except KeyError as e:
            raise ValueError(f"config.paths is missing required key: {e}") from e
        matches_debug_path       = _optional_path(paths_cfg.get("matches_debug"))
        rejected_dataset_1_path  = _optional_path(paths_cfg.get("rejected_dataset_1"))
        rejected_dataset_2_path  = _optional_path(paths_cfg.get("rejected_dataset_2"))
        unmatched_dataset_1_path = _optional_path(paths_cfg.get("unmatched_dataset_1"))
        unmatched_dataset_2_path = _optional_path(paths_cfg.get("unmatched_dataset_2"))

        # ----- Logging -----
        log_level = (raw.get("logging", {}) or {}).get("level", "INFO").upper()
        if log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(f"Unknown logging.level {log_level!r}")

        # ----- Scoring (strategy, weights, similarity, field_mappings) -----
        strategy = s["strategy"]
        if strategy not in ("per_field", "concatenated"):
            raise ValueError(f"Unknown strategy: {strategy!r}")

        weights_by_strategy = {}
        sim_by_strategy = {}
        mappings_by_strategy = {}
        for strat in ("per_field", "concatenated"):
            block = s[strat]
            weights    = {k: float(v) for k, v in block["weights"].items()}
            similarity = dict(block.get("similarity", {}))
            mappings   = dict(block.get("field_mappings", {}))

            # Validate similarity-function names exist in the registry.
            for field, name in similarity.items():
                if name not in SIMILARITY_FUNCTIONS:
                    raise ValueError(
                        f"Unknown similarity {name!r} for {strat}.{field}. "
                        f"Available: {sorted(SIMILARITY_FUNCTIONS)}"
                    )

            # Validate weights/similarity/field_mappings agree on the set of fields.
            wkeys, skeys, mkeys = set(weights), set(similarity), set(mappings)
            if not (wkeys == skeys == mkeys):
                raise ValueError(
                    f"scoring.{strat}: weights, similarity, and field_mappings "
                    f"must cover the same set of fields. "
                    f"weights={sorted(wkeys)}, similarity={sorted(skeys)}, "
                    f"field_mappings={sorted(mkeys)}"
                )

            # Validate field_mappings shape: each field has both `combined` and `split` lists.
            for field, mapping in mappings.items():
                for side in ("combined", "split"):
                    cols = mapping.get(side, [])
                    if not isinstance(cols, list) or not cols:
                        raise ValueError(
                            f"scoring.{strat}.field_mappings.{field}.{side} must be a "
                            f"non-empty list of column names; got {cols!r}"
                        )

            weights_by_strategy[strat]  = weights
            sim_by_strategy[strat]      = similarity
            mappings_by_strategy[strat] = mappings

        # ----- Assignment -----
        assignment_cfg = raw["assignment"]
        assignment_method = assignment_cfg["method"]
        if assignment_method not in assignment.METHODS:
            raise ValueError(
                f"Unknown assignment method {assignment_method!r}. "
                f"Available: {sorted(assignment.METHODS)}"
            )
        anchor_by_min_id_on = assignment_cfg.get("anchor_by_min_id_on")
        if anchor_by_min_id_on not in (None, "combined", "split"):
            raise ValueError(
                f"assignment.anchor_by_min_id_on must be null, 'combined', or "
                f"'split'; got {anchor_by_min_id_on!r}"
            )

        # ----- Blocking -----
        blocking_cfg = raw.get("blocking", {}) or {}
        blocking_enable_above_rows = int(blocking_cfg.get("enable_above_rows", 1000))
        blocking_keys = tuple(blocking_cfg.get("keys", []))

        # ----- Ingestion -----
        ingestion_cfg = raw.get("ingestion", {}) or {}
        ingestion_mode = ingestion_cfg.get("mode", "iterative")
        if ingestion_mode not in ingestion.CHECK_MODES:
            raise ValueError(
                f"Unknown ingestion.mode {ingestion_mode!r}. "
                f"Available: {sorted(ingestion.CHECK_MODES)}"
            )
        required_cfg = ingestion_cfg.get("required_fields", {}) or {}
        unique_cfg   = ingestion_cfg.get("unique_fields", {}) or {}

        return cls(
            dataset_1_path=dataset_1_path,
            dataset_2_path=dataset_2_path,
            matches_path=matches_path,
            matches_debug_path=matches_debug_path,
            rejected_dataset_1_path=rejected_dataset_1_path,
            rejected_dataset_2_path=rejected_dataset_2_path,
            unmatched_dataset_1_path=unmatched_dataset_1_path,
            unmatched_dataset_2_path=unmatched_dataset_2_path,
            log_level=log_level,
            strategy=strategy,
            weights_by_strategy=weights_by_strategy,
            similarity_by_strategy=sim_by_strategy,
            field_mappings_by_strategy=mappings_by_strategy,
            match_threshold=float(s["match_threshold"]),
            assignment_method=assignment_method,
            anchor_by_min_id_on=anchor_by_min_id_on,
            blocking_enable_above_rows=blocking_enable_above_rows,
            blocking_keys=blocking_keys,
            required_fields_combined=tuple(required_cfg.get("combined", [])),
            required_fields_split=tuple(required_cfg.get("split", [])),
            unique_fields_combined=tuple(unique_cfg.get("combined", [])),
            unique_fields_split=tuple(unique_cfg.get("split", [])),
            ingestion_mode=ingestion_mode,
            business_suffixes=frozenset(n["business_suffixes"]),
            street_abbreviations=dict(n["street_abbreviations"]),
            schema_combined=_validate_schema(raw["schema"]["combined"], "combined"),
            schema_split=_validate_schema(raw["schema"]["split"], "split"),
            composites=raw["schema"].get("composites", {}) or {},
        )


def _validate_schema(schema: dict, side: str) -> dict:
    """Sanity-check a schema entry early so errors are clear, not cryptic."""
    if "columns" not in schema:
        raise ValueError(f"schema.{side}: missing required 'columns' key")
    valid_fns = {"name", "text", "zip5"}
    for output_col, spec in schema["columns"].items():
        if "source" not in spec or "fn" not in spec:
            raise ValueError(
                f"schema.{side}.columns.{output_col}: each entry needs 'source' and 'fn'"
            )
        if spec["fn"] not in valid_fns:
            raise ValueError(
                f"schema.{side}.columns.{output_col}.fn={spec['fn']!r} invalid. "
                f"Available: {sorted(valid_fns)}"
            )
    return schema


def _optional_path(value) -> Path | None:
    """YAML may set an optional path to a string, null, or omit it. All
    of those except a non-empty string mean 'skip this output'."""
    if value is None or value == "" or value == "null":
        return None
    return Path(value)


# ===========================================================================
# Normalization
# ===========================================================================

class Normalizer:
    """Holds the rules needed to normalize names, addresses, and zip codes."""

    def __init__(self, config: Config):
        self.suffixes = config.business_suffixes
        self.abbrs = config.street_abbreviations

    @staticmethod
    def _basic_clean(s) -> str:
        if pd.isna(s):
            return ""
        s = str(s).upper()
        s = re.sub(r"[^A-Z0-9\s]", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    def name(self, s) -> str:
        s = self._basic_clean(s)
        if not s:
            return ""
        return " ".join(
            t for t in s.split()
            if t not in self.suffixes and not t.isdigit()
        )

    def text(self, s) -> str:
        """Address-style text: uppercase, depunctuated, abbreviations expanded."""
        s = self._basic_clean(s)
        if not s:
            return ""
        return " ".join(self.abbrs.get(t, t) for t in s.split())

    @staticmethod
    def zip5(z) -> str:
        if pd.isna(z):
            return ""
        return re.sub(r"\D", "", str(z))[:5]


# ===========================================================================
# Address parsing for dataset_1
# ===========================================================================

_ADDR_FULL      = re.compile(r"^(.+),\s*(.+),\s*([A-Z]{2})\s+(\d{5})(?:-\d{4})?\s*$")
_ADDR_NO_STREET = re.compile(r"^(.+),\s*([A-Z]{2})\s+(\d{5})(?:-\d{4})?\s*$")


def parse_combined_address(s: str) -> tuple[str, str, str, str]:
    """Return (street, city, state, zip5). Empty string for any missing piece."""
    if pd.isna(s):
        return "", "", "", ""
    s = str(s).strip()
    m = _ADDR_FULL.match(s)
    if m:
        return m.group(1).strip(), m.group(2).strip(), m.group(3), m.group(4)
    m = _ADDR_NO_STREET.match(s)
    if m:
        return "", m.group(1).strip(), m.group(2), m.group(3)
    return "", "", "", ""


# ===========================================================================
# Loading. Both prep functions produce all columns either strategy needs.
# ===========================================================================

def prep_dataset(df: pd.DataFrame, norm: "Normalizer", schema: dict, composites: dict) -> pd.DataFrame:
    """
    Generic per-side preparation. Replaces the old hardcoded
    `prep_combined` / `prep_split`. Schema-driven: adding a new CSV
    column means adding one line in `schema.<side>.columns`, no code
    change.

    Steps:
      1. (optional) Parse a free-form combined-address column into raw parts
      2. Apply per-column normalizers per `schema.columns`
      3. Build composite columns per `composites`
    """
    df = _maybe_parse_combined_address(df, schema)
    out = _normalize_per_schema(df, schema, norm)
    _apply_composites(out, composites)
    return out


def _maybe_parse_combined_address(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """If schema declares a combined-address column, parse it into raw
    parts. The raw parts (_street_raw / _city_raw / _state_raw / _zip_raw)
    become available as sources for the normalization step."""
    addr_col = schema.get("parse_combined_address_from")
    if not addr_col:
        return df
    parsed = df[addr_col].map(parse_combined_address)
    streets, cities, states, zips = zip(*parsed) if len(parsed) else ([], [], [], [])
    df = df.copy()
    df["_street_raw"] = pd.Series(streets, index=df.index)
    df["_city_raw"]   = pd.Series(cities,  index=df.index)
    df["_state_raw"] = pd.Series(states,  index=df.index)
    df["_zip_raw"]    = pd.Series(zips,    index=df.index)
    return df


def _normalize_per_schema(df: pd.DataFrame, schema: dict, norm: "Normalizer") -> pd.DataFrame:
    """Apply schema.columns rules in declared order. A source can be:
      - a raw column in `df`
      - a previously-computed column in `out` (must appear earlier in
        schema.columns)
    """
    out = pd.DataFrame({"id": df["id"].astype(str)})

    def _resolve(col_name: str) -> pd.Series:
        if col_name in out.columns:
            return out[col_name]
        if col_name in df.columns:
            return df[col_name]
        raise ValueError(
            f"schema source {col_name!r} not found. Available raw: {list(df.columns)}, "
            f"available previously-computed: {list(out.columns)}"
        )

    for output_col, spec in schema["columns"].items():
        source = spec["source"]
        fn = getattr(norm, spec["fn"])         # norm.name, norm.text, norm.zip5

        if isinstance(source, list):
            sep = spec.get("sep", " ")
            parts = [_resolve(c).fillna("").astype(str) for c in source]
            joined = parts[0]
            for p in parts[1:]:
                joined = joined + sep + p
            out[output_col] = joined.map(fn)
        else:
            out[output_col] = _resolve(source).map(fn)

    return out


def _apply_composites(out: pd.DataFrame, composites: dict) -> None:
    """Build composite columns from declarative specs. Each composite
    is a list of parts; parts can be literals or column references with
    optional prefix/suffix slicing."""
    for name, spec in composites.items():
        parts = spec.get("parts", [])
        result = pd.Series([""] * len(out), index=out.index, dtype=object)
        for inp in parts:
            if isinstance(inp, str):
                # Literal
                result = result + inp
            elif isinstance(inp, dict) and "col" in inp:
                col = out[inp["col"]].astype(str)
                if "prefix" in inp:
                    col = col.str[:int(inp["prefix"])]
                elif "suffix" in inp:
                    col = col.str[-int(inp["suffix"]):]
                result = result + col
            else:
                raise ValueError(
                    f"composites.{name}.parts contains unrecognized entry: {inp!r}. "
                    f"Use a string literal or a dict with 'col' (and optional prefix/suffix)."
                )
        out[name] = result


# ===========================================================================
# Scoring. ONE generic function drives all fields, parameterized by:
#   - which normalized columns to compare on each side (from config.field_mappings)
#   - which similarity function to apply           (from config.similarity)
# Returns an NxM matrix of 0..100 scores; np.nan means "could not compare".
# ===========================================================================

def score_field(left: pd.DataFrame, right: pd.DataFrame,
                left_cols: list, right_cols: list,
                similarity_fn) -> np.ndarray:
    """
    Pairwise similarity over the listed columns on each side.

    result[i, j] is the MAX similarity across every (left_col, right_col)
    pair for that row pair. A row contributes nothing on a side when all
    its listed columns are empty; the cell is NaN in that case so the
    NaN-aware combiner can redistribute the weight.

    The single-column-on-each-side case (e.g. street vs street) and the
    many-on-one-side case (e.g. name in three slots on dataset_2 vs one
    on dataset_1) share the same code path.
    """
    _check_cols(left, left_cols, "left")
    _check_cols(right, right_cols, "right")

    n, m = len(left), len(right)
    out = np.full((n, m), np.nan, dtype=float)

    # Pre-fetch column values as plain lists; iloc/loc inside the inner
    # loop is much slower.
    left_col_vals  = [left[c].tolist()  for c in left_cols]
    right_col_vals = [right[c].tolist() for c in right_cols]

    for i in range(n):
        left_candidates = [vals[i] for vals in left_col_vals if vals[i]]
        if not left_candidates:
            continue
        for j in range(m):
            right_candidates = [vals[j] for vals in right_col_vals if vals[j]]
            if not right_candidates:
                continue
            out[i, j] = max(
                similarity_fn(l, r)
                for l in left_candidates
                for r in right_candidates
            )
    return out


def _check_cols(df: pd.DataFrame, cols: list, side: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{side} DataFrame is missing columns required by field_mappings: "
            f"{missing}. Available: {list(df.columns)}"
        )


# ===========================================================================
# Combination
# ===========================================================================

def combine_scores(score_matrices: dict, weights: dict) -> np.ndarray:
    """
    NaN-aware weighted average. For each cell, sum the weighted scores
    over the fields that DID produce a comparison, divided by the sum
    of those fields' weights. Missing fields neither help nor hurt.
    """
    fields = list(weights.keys())
    stacked = np.stack([score_matrices[f] for f in fields])
    w = np.array([weights[f] for f in fields]).reshape(-1, 1, 1)

    valid = ~np.isnan(stacked)
    safe = np.where(valid, stacked, 0.0)
    numerator = (safe * w).sum(axis=0)
    denominator = (valid * w).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / denominator, 0.0)


# ===========================================================================
# Strategy dispatch
# ===========================================================================

def compute_field_scores(config: Config, split: pd.DataFrame, combined: pd.DataFrame) -> dict:
    """
    Score every field declared in `config.field_mappings`.

    For each field the config tells us:
      - which normalized columns to compare on each side (split/combined)
      - which similarity function to use
    so this becomes a uniform loop. To add a new field, declare it in the
    YAML (`weights`, `similarity`, and `field_mappings` per strategy) and
    make sure prep_combined / prep_split produce the referenced columns.
    """
    result = {}
    for field, mapping in config.field_mappings.items():
        similarity_fn = SIMILARITY_FUNCTIONS[config.similarity[field]]
        result[field] = score_field(
            left=split,
            right=combined,
            left_cols=mapping["split"],
            right_cols=mapping["combined"],
            similarity_fn=similarity_fn,
        )
    return result


# ===========================================================================
# Orchestration
# ===========================================================================

def score_and_assign(split: pd.DataFrame, combined: pd.DataFrame, config: Config):
    """
    Run the full scoring + assignment pipeline on a (sub)set of records and
    return (pairs, scores_matrix, field_scores). Used both for the full
    dataset and per-block when blocking is enabled.
    """
    field_scores = compute_field_scores(config, split, combined)
    scores = combine_scores(field_scores, config.weights)
    pairs = assignment.assign(config.assignment_method, scores, config.match_threshold)
    return pairs, scores, field_scores


def _run_full(split: pd.DataFrame, combined: pd.DataFrame, config: Config):
    """Score every pair. No blocking. Quadratic, fine for small datasets."""
    pairs, scores, field_scores = score_and_assign(split, combined, config)
    pair_count = scores.size
    return pairs, scores, field_scores, pair_count


def _run_blocked(split: pd.DataFrame, combined: pd.DataFrame, config: Config):
    """
    Run the SAME `score_and_assign` logic as `_run_full`, but on each
    blocking group's chunk instead of the full DataFrame. Multi-key
    blocking iterates each key separately; a pair that appears under
    multiple keys keeps its best score. Finally, enforce the assignment
    method's cardinality across all the blocks.
    """
    pair_scores: dict = {}             # (global_i, global_j) -> max score
    field_score_per_pair: dict = {}    # (global_i, global_j) -> {field: score}
    pair_count = 0

    for key in config.blocking_keys:
        for _value, split_chunk, combined_chunk in blocking.iter_chunks(split, combined, key):
            pair_count += len(split_chunk) * len(combined_chunk)

            # Identical to the call inside _run_full — just on a chunk.
            chunk_pairs, _scores, chunk_field_scores = score_and_assign(
                split_chunk.reset_index(drop=True),
                combined_chunk.reset_index(drop=True),
                config,
            )

            # Translate local (positional) indices back to global ones.
            split_global    = split_chunk.index.to_numpy()
            combined_global = combined_chunk.index.to_numpy()

            for li, lj, sc in chunk_pairs:
                gi, gj = int(split_global[li]), int(combined_global[lj])
                if (gi, gj) not in pair_scores or sc > pair_scores[(gi, gj)]:
                    pair_scores[(gi, gj)] = sc
                    field_score_per_pair[(gi, gj)] = {
                        fname: float(mat[li, lj]) if not np.isnan(mat[li, lj]) else np.nan
                        for fname, mat in chunk_field_scores.items()
                    }

    final = assignment.apply_global_cardinality(pair_scores, config.assignment_method)
    return final, field_score_per_pair, pair_count


def run(config: Config):
    """
    Top-level pipeline. Everything it needs lives in `config`.
    Each numbered stage delegates to a focused helper below.
    """
    start = time.time()
    logger.info("Pipeline start: strategy=%s, assignment=%s, blocking_threshold=%d rows",
                config.strategy, config.assignment_method,
                config.blocking_enable_above_rows)

    norm = Normalizer(config)

    # Stage 1: Read raw data. Kept inline because (a) it's two lines and
    # (b) the file format is the caller's choice — swap pd.read_csv for
    # pd.read_parquet, pd.read_json, or a SQL query and nothing downstream
    # cares.
    logger.info("Reading dataset_1 from %s", config.dataset_1_path)
    combined_raw = pd.read_csv(config.dataset_1_path)
    logger.info("Reading dataset_2 from %s", config.dataset_2_path)
    split_raw = pd.read_csv(config.dataset_2_path)

    # Stage 2: Validate and dedupe.
    ingest_combined, ingest_split = _ingest(combined_raw, split_raw, config)

    # Stage 3: Persist rejection logs (early, so they exist even if a
    # later stage crashes).
    _write_rejection_logs(ingest_combined, ingest_split, config)

    # Stage 4: Normalize.
    split, combined = _normalize(ingest_combined, ingest_split, norm, config)

    # Stage 5: Score + assign matches.
    result = _match(split, combined, config)

    # Stage 6: Write matched output CSVs.
    _write_outputs(result, split, combined, config)

    # Stage 7: Write unmatched records — rows that passed ingestion but
    # didn't end up in any pair. Complementary to matches.csv; together
    # they account for every record that survived ingestion.
    _write_unmatched(result, split, combined, config)

    logger.info("Pipeline finished in %.2fs", time.time() - start)


# ===========================================================================
# Stage helpers. Each is a method called by `run()`. To change a stage's
# behavior, modify the corresponding helper without touching the others.
# To add a new stage, write a helper here and add one line to `run()`.
# ===========================================================================

@dataclass
class MatchResult:
    """Output of the matching stage. Passed into `_write_outputs`."""
    pairs: list                 # list of (split_idx, combined_idx, score) tuples
    field_score_getter: object  # callable: (i, j) -> {field_name: score}
    pair_count_scored: int      # how many pairs the matcher actually scored
    pair_count_naive: int       # what full N*M would have been (for the log)


def _ingest(combined_raw: pd.DataFrame, split_raw: pd.DataFrame,
            config: Config) -> tuple[ingestion.IngestResult, ingestion.IngestResult]:
    """Stage 2: validate + dedupe both datasets. The ingestion module
    handles logging of per-source kept/rejected counts."""
    ingest_combined = ingestion.apply_checks(
        combined_raw, "dataset_1",
        required_fields=config.required_fields_combined,
        unique_fields=config.unique_fields_combined,
        mode=config.ingestion_mode,
    )
    ingest_split = ingestion.apply_checks(
        split_raw, "dataset_2",
        required_fields=config.required_fields_split,
        unique_fields=config.unique_fields_split,
        mode=config.ingestion_mode,
    )
    return ingest_combined, ingest_split


def _write_rejection_logs(
    ingest_combined: ingestion.IngestResult,
    ingest_split: ingestion.IngestResult,
    config: Config,
) -> None:
    """Stage 3: persist rejection logs to paths from config. Skipped
    per-side when the corresponding config path is null."""
    if config.rejected_dataset_1_path is not None:
        ingestion.write(ingest_combined.rejected, config.rejected_dataset_1_path)
    if config.rejected_dataset_2_path is not None:
        ingestion.write(ingest_split.rejected, config.rejected_dataset_2_path)


def _normalize(
    ingest_combined: ingestion.IngestResult,
    ingest_split: ingestion.IngestResult,
    norm: Normalizer,
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage 4: schema-driven normalization. Returns (split, combined)
    in the order used downstream by `_match`."""
    logger.info("Normalizing %d + %d records",
                len(ingest_combined.kept), len(ingest_split.kept))
    combined = prep_dataset(ingest_combined.kept, norm, config.schema_combined, config.composites)
    split    = prep_dataset(ingest_split.kept,    norm, config.schema_split,    config.composites)
    return split, combined


def _apply_min_id_anchor(
    pairs: list,
    split_ids: list,
    combined_ids: list,
    parent_side: str,
) -> list:
    """
    Collapse the input pairs so each non-parent record keeps only ONE
    partner: the one whose id on the parent side is smallest.

    Used to break ties left over by `threshold_only` (or any method
    that allows multiple partners) using a deterministic, score-free
    rule. The smallest-id partner becomes the canonical "parent."

    parent_side='combined'  -> each split (row) keeps its min-combined-id partner
    parent_side='split'     -> each combined (col) keeps its min-split-id partner
    """
    if parent_side == "combined":
        # Group by row (split index); keep the column whose combined id is smallest.
        best_per_row = {}
        for i, j, sc in pairs:
            current = best_per_row.get(i)
            if current is None or combined_ids[j] < combined_ids[current[0]]:
                best_per_row[i] = (j, sc)
        return [(i, j, sc) for i, (j, sc) in best_per_row.items()]

    if parent_side == "split":
        # Group by column (combined index); keep the row whose split id is smallest.
        best_per_col = {}
        for i, j, sc in pairs:
            current = best_per_col.get(j)
            if current is None or split_ids[i] < split_ids[current[0]]:
                best_per_col[j] = (i, sc)
        return [(i, j, sc) for j, (i, sc) in best_per_col.items()]

    raise ValueError(
        f"parent_side must be 'combined' or 'split'; got {parent_side!r}"
    )


def _match(split: pd.DataFrame, combined: pd.DataFrame,
           config: Config) -> MatchResult:
    """Stage 5: decide whether to block (by size), then score + assign.

    All of the blocking-vs-full logic lives here so `run()` doesn't need
    to know how matching is done internally — it just gets a MatchResult.
    """
    naive_pair_count = len(split) * len(combined)
    max_rows = max(len(split), len(combined))
    use_blocking = (
        bool(config.blocking_keys)
        and max_rows > config.blocking_enable_above_rows
    )

    if use_blocking:
        logger.info(
            "Blocking ENABLED: max_rows=%d > threshold=%d, keys=%s",
            max_rows, config.blocking_enable_above_rows, list(config.blocking_keys),
        )
        pairs, field_score_per_pair, scored_count = _run_blocked(split, combined, config)
        getter = lambda i, j: field_score_per_pair.get((i, j), {})
    else:
        reason = (
            "no blocking keys configured"
            if not config.blocking_keys
            else f"max_rows={max_rows} <= threshold={config.blocking_enable_above_rows}"
        )
        logger.info("Blocking SKIPPED (%s): scoring full %d x %d matrix",
                    reason, len(split), len(combined))
        chunk_pairs, _scores, field_scores, scored_count = _run_full(split, combined, config)
        pairs = chunk_pairs
        getter = lambda i, j: {
            fname: float(mat[i, j]) if not np.isnan(mat[i, j]) else np.nan
            for fname, mat in field_scores.items()
        }

    # Optional min-ID anchoring. Applied AFTER the cardinality-aware
    # assignment, so methods like threshold_only that allow multiple
    # parents per child can be collapsed to a single deterministic
    # parent. With 1:1 methods this is a no-op (no ambiguity to break).
    if config.anchor_by_min_id_on is not None:
        before = len(pairs)
        pairs = _apply_min_id_anchor(
            pairs,
            split_ids=split["id"].tolist(),
            combined_ids=combined["id"].tolist(),
            parent_side=config.anchor_by_min_id_on,
        )
        logger.info("Anchor by min id on %s: %d -> %d pairs",
                    config.anchor_by_min_id_on, before, len(pairs))

    reduction = (1 - scored_count / naive_pair_count) * 100 if naive_pair_count else 0
    logger.info("Scored %d of %d candidate pairs (%.1f%% reduction)",
                scored_count, naive_pair_count, reduction)
    logger.info("Matched %d of %d possible pairs (threshold=%.1f)",
                len(pairs), min(len(split), len(combined)), config.match_threshold)

    return MatchResult(
        pairs=pairs,
        field_score_getter=getter,
        pair_count_scored=scored_count,
        pair_count_naive=naive_pair_count,
    )


def _write_outputs(
    result: MatchResult,
    split: pd.DataFrame, combined: pd.DataFrame,
    config: Config,
) -> None:
    """Stage 6: write outputs to paths from config. Debug CSV skipped if null."""
    out_rows = [
        {"id_1": split.iloc[i]["id"], "id_2": combined.iloc[j]["id"]}
        for i, j, _ in result.pairs
    ]
    pd.DataFrame(out_rows, columns=["id_1", "id_2"]).to_csv(config.matches_path, index=False)
    logger.info("Wrote matches: %s", config.matches_path)

    if config.matches_debug_path is None:
        return

    rows = []
    for i, j, score in result.pairs:
        row = {
            "id_1": split.iloc[i]["id"],
            "id_2": combined.iloc[j]["id"],
            "combined": round(score, 1),
        }
        for fname, v in result.field_score_getter(i, j).items():
            row[f"{fname}_score"] = None if (isinstance(v, float) and np.isnan(v)) else round(v, 1)
        rows.append(row)
    pd.DataFrame(rows).to_csv(config.matches_debug_path, index=False)
    logger.info("Wrote debug view: %s", config.matches_debug_path)


def _write_unmatched(
    result: MatchResult,
    split: pd.DataFrame, combined: pd.DataFrame,
    config: Config,
) -> None:
    """Stage 7: write the rows that DIDN'T end up in any matched pair.

    A row can be unmatched for two reasons:
      - its best candidate scored below `match_threshold`, or
      - in a 1:1 assignment (hungarian/greedy_mutual/mutual_best), its
        best candidate was already taken by a higher-scoring pair.

    Together with matches.csv this gives a complete accounting of every
    post-ingestion row: it's either matched, or it's in unmatched_*.csv.
    """
    matched_split    = {i for i, _, _ in result.pairs}
    matched_combined = {j for _, j, _ in result.pairs}

    if config.unmatched_dataset_2_path is not None:
        unmatched_pos = [i for i in range(len(split)) if i not in matched_split]
        split.iloc[unmatched_pos].to_csv(config.unmatched_dataset_2_path, index=False)
        logger.info("Wrote unmatched dataset_2: %s (%d rows)",
                    config.unmatched_dataset_2_path, len(unmatched_pos))

    if config.unmatched_dataset_1_path is not None:
        unmatched_pos = [j for j in range(len(combined)) if j not in matched_combined]
        combined.iloc[unmatched_pos].to_csv(config.unmatched_dataset_1_path, index=False)
        logger.info("Wrote unmatched dataset_1: %s (%d rows)",
                    config.unmatched_dataset_1_path, len(unmatched_pos))

    # Sanity-check summary: every post-ingestion record is accounted for.
    logger.info(
        "Match summary: %d paired | dataset_1: %d matched + %d unmatched = %d | "
        "dataset_2: %d matched + %d unmatched = %d",
        len(result.pairs),
        len(matched_combined), len(combined) - len(matched_combined), len(combined),
        len(matched_split),    len(split)    - len(matched_split),    len(split),
    )


def main():
    p = argparse.ArgumentParser(description="Match business records by name + address.")
    p.add_argument("--config", default="config.yml",
                   help="Path to the YAML config (all other settings live there).")
    args = p.parse_args()

    config = Config.from_yaml(Path(args.config))

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s [%(levelname)-7s] %(name)-15s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config.matches_path.parent.mkdir(parents=True, exist_ok=True)
    run(config)


if __name__ == "__main__":
    main()
