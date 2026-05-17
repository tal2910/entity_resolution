"""
Blocking: group records by a column value so we only score pairs that
share that value.

The orchestrator iterates groups and runs the SAME scoring/assignment
logic it would run on the full dataset. There's no per-block control
flow that needs to live here — this module just yields chunks.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def iter_chunks(left: pd.DataFrame, right: pd.DataFrame, key: str):
    """
    For each value of `key` present on BOTH sides (excluding empty
    strings), yield `(value, left_chunk, right_chunk)`.

    Chunks preserve their ORIGINAL DataFrame index so the caller can
    translate local (positional) pair indices back to global row indices
    via the chunk's `.index`.
    """
    for side, df in (("left", left), ("right", right)):
        if key not in df.columns:
            raise KeyError(
                f"Blocking key {key!r} not found in {side} DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

    shared = sorted(set(left[key]) & set(right[key]) - {""})
    logger.info("Blocking on '%s': %d shared values", key, len(shared))

    for value in shared:
        yield value, left[left[key] == value], right[right[key] == value]
