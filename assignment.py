"""
Pluggable assignment methods for selecting pairs from a score matrix.

Cardinality                     Methods
------------------------------- -----------------------------------------
1:1     (each side ≤ 1 partner) hungarian, greedy_mutual, mutual_best
1:many  (rows ≤ 1, cols any)    one_per_row
many:1  (rows any, cols ≤ 1)    one_per_col
many:many (no constraint)       threshold_only

`assign(name, scores, threshold)` runs one method on a score matrix.
After blocking, the orchestrator calls `apply_global_cardinality(...)`
to reconcile per-block results so the same cardinality rule holds
GLOBALLY, not just per block.
"""

from typing import Callable
import logging

import numpy as np
from scipy.optimize import linear_sum_assignment


logger = logging.getLogger(__name__)

PairList = list[tuple[int, int, float]]


# ---------------------------------------------------------------------------
# 1:1 methods
# ---------------------------------------------------------------------------

def hungarian(scores: np.ndarray, threshold: float) -> PairList:
    """Globally optimal 1:1 via Jonker-Volgenant. O(n^3)."""
    row_idx, col_idx = linear_sum_assignment(-scores)
    return [
        (int(r), int(c), float(scores[r, c]))
        for r, c in zip(row_idx, col_idx)
        if scores[r, c] >= threshold
    ]


def greedy_mutual(scores: np.ndarray, threshold: float) -> PairList:
    """Pick highest pair, lock row+col, repeat. Identical to Hungarian
    when score contrast is high; much faster."""
    n, m = scores.shape
    if n == 0 or m == 0:
        return []
    flat = np.argsort(scores, axis=None)[::-1]
    used_rows, used_cols = set(), set()
    pairs: PairList = []
    for idx in flat:
        i, j = int(idx // m), int(idx % m)
        s = float(scores[i, j])
        if s < threshold:
            break
        if i in used_rows or j in used_cols:
            continue
        used_rows.add(i); used_cols.add(j)
        pairs.append((i, j, s))
    return pairs


def mutual_best(scores: np.ndarray, threshold: float) -> PairList:
    """High precision: pairs that are mutually argmax."""
    if scores.size == 0:
        return []
    best_col_per_row = scores.argmax(axis=1)
    best_row_per_col = scores.argmax(axis=0)
    pairs: PairList = []
    for i, j in enumerate(best_col_per_row):
        j = int(j)
        if best_row_per_col[j] == i and scores[i, j] >= threshold:
            pairs.append((i, j, float(scores[i, j])))
    return pairs


# ---------------------------------------------------------------------------
# many:many
# ---------------------------------------------------------------------------

def threshold_only(scores: np.ndarray, threshold: float) -> PairList:
    """No cardinality constraint: every pair >= threshold."""
    rows, cols = np.where(scores >= threshold)
    return [(int(i), int(j), float(scores[i, j])) for i, j in zip(rows, cols)]


# ---------------------------------------------------------------------------
# 1:many / many:1
# ---------------------------------------------------------------------------

def one_per_row(scores: np.ndarray, threshold: float) -> PairList:
    """
    1:many. Each ROW gets at most one match (its argmax column, if
    above threshold). Columns may appear multiple times — multiple
    rows can map to the same column.

    Use when rows are the 'many' side and cols are the 'one'. Example:
    rows are branches, cols are HQs. Each branch belongs to one HQ;
    one HQ has many branches.
    """
    if scores.size == 0:
        return []
    safe = np.where(np.isnan(scores), -np.inf, scores)
    best_col = safe.argmax(axis=1)
    pairs: PairList = []
    for i in range(scores.shape[0]):
        j = int(best_col[i])
        if safe[i, j] >= threshold:
            pairs.append((i, j, float(scores[i, j])))
    return pairs


def one_per_col(scores: np.ndarray, threshold: float) -> PairList:
    """many:1. Mirror of one_per_row: each column gets at most one
    match. Rows may appear multiple times."""
    if scores.size == 0:
        return []
    safe = np.where(np.isnan(scores), -np.inf, scores)
    best_row = safe.argmax(axis=0)
    pairs: PairList = []
    for j in range(scores.shape[1]):
        i = int(best_row[j])
        if safe[i, j] >= threshold:
            pairs.append((i, j, float(scores[i, j])))
    return pairs


# ---------------------------------------------------------------------------
# Registry + cardinality metadata
# ---------------------------------------------------------------------------

METHODS: dict[str, Callable[[np.ndarray, float], PairList]] = {
    "hungarian":      hungarian,
    "greedy_mutual":  greedy_mutual,
    "mutual_best":    mutual_best,
    "threshold_only": threshold_only,
    "one_per_row":    one_per_row,
    "one_per_col":    one_per_col,
}

# What cardinality each method enforces. Used by apply_global_cardinality
# below so the blocking-dedup pass applies the same constraint globally
# (not just per-block, which would let pairs sneak past).
METHOD_CARDINALITY: dict[str, str] = {
    "hungarian":      "one_to_one",
    "greedy_mutual":  "one_to_one",
    "mutual_best":    "one_to_one",
    "threshold_only": "many_to_many",
    "one_per_row":    "row_unique",
    "one_per_col":    "col_unique",
}


def assign(method: str, scores: np.ndarray, threshold: float) -> PairList:
    """Dispatch to the named assignment method, with validation."""
    if method not in METHODS:
        raise ValueError(
            f"Unknown assignment method {method!r}. Available: {sorted(METHODS)}"
        )
    pairs = METHODS[method](scores, threshold)
    logger.debug("assignment.%s on %s matrix returned %d pairs (threshold=%s)",
                 method, scores.shape, len(pairs), threshold)
    return pairs


def apply_global_cardinality(pair_scores: dict, method: str) -> PairList:
    """
    Reduce a `{(i, j): score}` dict to a `[(i, j, score), ...]` list
    that respects `method`'s cardinality GLOBALLY.

    Used after blocking. Per-block assignment may produce pairs that,
    when unioned, violate the row/col-uniqueness invariant. A single
    greedy descending-score sweep restores it.
    """
    cardinality = METHOD_CARDINALITY[method]
    sorted_pairs = sorted(pair_scores.items(), key=lambda kv: -kv[1])

    if cardinality == "many_to_many":
        return [(i, j, sc) for (i, j), sc in sorted_pairs]

    used_i: set = set()
    used_j: set = set()
    out: PairList = []

    for (i, j), sc in sorted_pairs:
        if cardinality == "one_to_one":
            if i in used_i or j in used_j:
                continue
        elif cardinality == "row_unique":
            if i in used_i:
                continue
        elif cardinality == "col_unique":
            if j in used_j:
                continue
        else:
            raise ValueError(f"Unknown cardinality {cardinality!r}")
        used_i.add(i)
        used_j.add(j)
        out.append((i, j, sc))
    return out
