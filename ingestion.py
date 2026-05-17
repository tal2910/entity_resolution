"""
Ingestion: validation and rejection logging.

Design
------

This module operates on pandas DataFrames, not on files. The caller is
responsible for reading the data into a DataFrame first — from CSV today,
Parquet, JSON, or a database query tomorrow. Ingestion does not care
about the source format.

Each check exists in TWO equivalent implementations:

    *_iterative   plain Python row-by-row loops. Easier to read, easier
                  to modify, easier to step through in a debugger.
                  This is the default and what you should reach for
                  when you're changing logic.

    *_vectorized  pandas mask / groupby operations. Roughly 2x faster
                  than iterative across typical sizes (verified by
                  benchmark up to 200k rows). The speedup is bounded
                  by the dict-building loop over rejected rows, which
                  both modes share. Reach for vectorized when iterative
                  becomes a noticeable wait, not as a routine default.

The two modes are guaranteed to produce identical kept and rejected
DataFrames. `apply_checks(..., mode=...)` picks which set runs.

Adding a new check
------------------
    1. Write a `check_yyy_iterative` function and a `check_yyy_vectorized`
       function with the same signature.
    2. Add both to the CHECK_MODES registry below.
    3. Add two lines to `apply_checks` invoking the new check.
"""

import logging
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


logger = logging.getLogger(__name__)


META_COLUMNS = ["_source", "_reason", "_kept_id"]


@dataclass
class IngestResult:
    kept: pd.DataFrame
    rejected: pd.DataFrame


# ===========================================================================
# Public API
# ===========================================================================

def apply_checks(
    df: pd.DataFrame,
    source_name: str,
    required_fields: Iterable[str] = (),
    unique_fields: Iterable[str] = (),
    mode: str = "iterative",
) -> IngestResult:
    """Run the ingestion checks over `df` and return kept + rejected DataFrames."""
    if mode not in CHECK_MODES:
        raise ValueError(
            f"Unknown ingestion mode {mode!r}. Available: {sorted(CHECK_MODES)}"
        )
    impls = CHECK_MODES[mode]
    logger.info("%s: %d rows to validate (mode=%s)", source_name, len(df), mode)

    keep = df
    rejected = []

    # Add or reorder checks here. Both modes must implement the same set.
    keep, new = impls["required_fields"](keep, source_name, list(required_fields))
    rejected.extend(new)

    keep, new = impls["unique_fields"](keep, source_name, list(unique_fields))
    rejected.extend(new)

    rejected_df = (
        pd.DataFrame(rejected)
        if rejected
        else _empty_rejection_df(df.columns)
    )

    if rejected:
        logger.warning("%s: %d rejected (%s)",
                       source_name, len(rejected), _count_string(rejected_df))
    logger.info("%s: %d kept, %d rejected", source_name, len(keep), len(rejected))

    return IngestResult(kept=keep.reset_index(drop=True), rejected=rejected_df)


def write(rejected: pd.DataFrame, path) -> None:
    """Write rejected rows to a CSV. Always write, even when empty,
    so an analyst can prove the check ran and found nothing."""
    rejected.to_csv(path, index=False)
    logger.info("Wrote rejection log: %s (%d rows)", path, len(rejected))


def summarize(rejected: pd.DataFrame, source_label: str) -> str:
    """One-line, counts-by-reason summary for ad-hoc logging."""
    if len(rejected) == 0:
        return f"{source_label}: 0 rejected"
    return f"{source_label}: {len(rejected)} rejected ({_count_string(rejected)})"


# ===========================================================================
# Check: required_fields
# ===========================================================================

def check_required_fields_iterative(
    df: pd.DataFrame, source_name: str, required_fields: list
) -> tuple[pd.DataFrame, list]:
    """Reject rows where any required field is missing or blank. Row-by-row."""
    if not required_fields:
        return df, []
    _assert_columns_exist(df, required_fields, source_name, "required_fields")

    keep_records = []
    rejected_records = []

    for record in df.to_dict("records"):
        missing_field = None
        for col in required_fields:
            if _is_blank(record[col]):
                missing_field = col
                break

        if missing_field is None:
            keep_records.append(record)
        else:
            rejected_records.append(_rejection(
                record, source_name, f"missing_{missing_field}", kept_id=None,
            ))

    return _records_to_df(keep_records, df.columns), rejected_records


def check_required_fields_vectorized(
    df: pd.DataFrame, source_name: str, required_fields: list
) -> tuple[pd.DataFrame, list]:
    """Same as iterative but uses pandas masks. Faster on large data."""
    if not required_fields:
        return df, []
    _assert_columns_exist(df, required_fields, source_name, "required_fields")

    # For each row, record the first field it was missing (or None if it passed).
    # Building this Series in one pass over the columns avoids per-row Python.
    first_failure = pd.Series([None] * len(df), index=df.index, dtype=object)
    for col in required_fields:
        is_blank = df[col].isna() | (df[col].astype(str).str.strip() == "")
        unflagged = first_failure.isna()
        first_failure = first_failure.where(~(is_blank & unflagged), col)

    keep_mask = first_failure.isna()
    rejected_records = [
        _rejection(
            df.loc[idx].to_dict(), source_name,
            f"missing_{first_failure[idx]}", kept_id=None,
        )
        for idx in df.index[~keep_mask]
    ]
    return df[keep_mask].reset_index(drop=True), rejected_records


# ===========================================================================
# Check: unique_fields
# ===========================================================================

def check_unique_fields_iterative(
    df: pd.DataFrame, source_name: str, unique_fields: list
) -> tuple[pd.DataFrame, list]:
    """Reject rows duplicating an earlier row on `unique_fields`. Dict-based."""
    if not unique_fields:
        return df, []
    _assert_columns_exist(df, unique_fields, source_name, "unique_fields")

    seen: dict = {}                # key tuple -> id of the first row to produce that key
    keep_records = []
    rejected_records = []
    reason = "duplicate_on_" + "_".join(unique_fields)

    for record in df.to_dict("records"):
        key = tuple(record[c] for c in unique_fields)
        if key in seen:
            rejected_records.append(_rejection(
                record, source_name, reason, kept_id=seen[key],
            ))
        else:
            seen[key] = record["id"]
            keep_records.append(record)

    return _records_to_df(keep_records, df.columns), rejected_records


def check_unique_fields_vectorized(
    df: pd.DataFrame, source_name: str, unique_fields: list
) -> tuple[pd.DataFrame, list]:
    """Same as iterative but uses df.duplicated() + a lookup dict. Faster."""
    if not unique_fields:
        return df, []
    _assert_columns_exist(df, unique_fields, source_name, "unique_fields")

    fields = list(unique_fields)
    dup_mask = df.duplicated(subset=fields, keep="first")
    if not dup_mask.any():
        return df, []

    # Lookup: tuple(field_values) -> id of the row that's the survivor for that key.
    first_per_group = df.drop_duplicates(subset=fields, keep="first")
    first_id_by_key = dict(zip(
        zip(*(first_per_group[c] for c in fields)),
        first_per_group["id"],
    ))

    reason = "duplicate_on_" + "_".join(fields)
    rejected_records = []
    for idx in df.index[dup_mask]:
        row = df.loc[idx]
        key = tuple(row[c] for c in fields)
        rejected_records.append(_rejection(
            row.to_dict(), source_name, reason, kept_id=first_id_by_key.get(key),
        ))

    return df[~dup_mask].reset_index(drop=True), rejected_records


# ===========================================================================
# Mode registry. To add a new check, fill in BOTH columns.
# To add a new mode (e.g. a Spark or Dask backend later), add another dict.
# ===========================================================================

CHECK_MODES = {
    "iterative": {
        "required_fields": check_required_fields_iterative,
        "unique_fields":   check_unique_fields_iterative,
    },
    "vectorized": {
        "required_fields": check_required_fields_vectorized,
        "unique_fields":   check_unique_fields_vectorized,
    },
}


# ===========================================================================
# Helpers
# ===========================================================================

def _is_blank(value) -> bool:
    """True if value is NaN, None, or a whitespace-only string."""
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _rejection(record: dict, source_name: str, reason: str, kept_id) -> dict:
    """Build a rejection-log entry: original row plus three metadata fields."""
    return {
        **record,
        "_source": source_name,
        "_reason": reason,
        "_kept_id": kept_id,
    }


def _assert_columns_exist(df: pd.DataFrame, columns: list,
                          source_name: str, setting_name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{source_name}: {setting_name} refers to columns missing from the "
            f"DataFrame: {missing}. Available columns: {list(df.columns)}"
        )


def _records_to_df(records: list, columns) -> pd.DataFrame:
    """Build a DataFrame from a list of record-dicts, preserving column order."""
    if not records:
        return pd.DataFrame(columns=list(columns))
    return pd.DataFrame(records)[list(columns)]


def _empty_rejection_df(input_columns) -> pd.DataFrame:
    return pd.DataFrame(columns=list(input_columns) + META_COLUMNS)


def _count_string(rejected_df: pd.DataFrame) -> str:
    counts = rejected_df["_reason"].value_counts()
    return ", ".join(f"{reason}={n}" for reason, n in counts.items())
