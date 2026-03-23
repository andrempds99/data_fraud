import logging

import pandas as pd
import numpy as np

from src.config import (
    TRAIN_CSV,
    TEST_CSV,
    TARGET,
    LABEL_COLS,
    NULL_TOKENS,
    BOOL_COLS,
    TYPE_PRIORITY,
    RESPONSE_FIXES,
    DROP_COLS,
)

logger = logging.getLogger(__name__)


# ── Loading ────────────────────────────────────────────────────────────────────

def load_data():
    """Load train/test CSVs, separate the target, and drop label columns."""
    train = pd.read_csv(TRAIN_CSV, encoding="utf-8", encoding_errors="replace")
    test = pd.read_csv(TEST_CSV, encoding="utf-8", encoding_errors="replace")

    y_train = train[TARGET].copy()
    train.drop(columns=LABEL_COLS, inplace=True)

    logger.info("Train shape: %s", train.shape)
    logger.info("Test  shape: %s", test.shape)
    logger.info("Fraud rate:  %.4f%%", y_train.mean() * 100)
    return train, test, y_train


# ── Null normalisation ─────────────────────────────────────────────────────────

def normalise_nulls(df):
    """Replace known null sentinel strings with np.nan."""
    df = df.copy()
    null_set = set(NULL_TOKENS)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip().str.lower()
        # Use boolean mask assignment instead of replace() to avoid the
        # pandas FutureWarning about silent downcasting in replace().
        mask = df[col].isin(null_set)
        df.loc[mask, col] = np.nan
    return df


# ── Deduplication ──────────────────────────────────────────────────────────────

def deduplicate(df, y=None):
    """Keep the most informative row per transactionid.

    Sort priority:
    1. Fraud-positive rows first (when ``y`` is provided), so chargebacks are
       never silently discarded by deduplication.
    2. Transaction-type priority: auth_capture > capture > auth.
    """
    df = df.copy()
    df["_type_priority"] = df["transactiontype"].map(TYPE_PRIORITY).fillna(99)

    if y is not None:
        # Embed label temporarily so fraud=1 rows survive de-duplication.
        df["_y"] = y.reindex(df.index).fillna(0).astype(int)
        df.sort_values(["_y", "_type_priority"], ascending=[False, True], inplace=True)
        y = y.loc[df.index]
    else:
        df.sort_values("_type_priority", inplace=True)

    mask = df.duplicated(subset="transactionid", keep="first")
    df = df[~mask]
    df.drop(columns="_type_priority", inplace=True)

    if y is not None:
        df.drop(columns="_y", inplace=True)
        y = y.loc[df.index]
        return df, y
    return df


# ── Encoding fixes ─────────────────────────────────────────────────────────────

def fix_response_codes(df):
    df = df.copy()
    df["responsecode"] = df["responsecode"].replace(RESPONSE_FIXES)
    return df


# ── Type casting ───────────────────────────────────────────────────────────────

def cast_types(df):
    """Convert boolean strings to int and parse timestamps."""
    df = df.copy()

    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].map({"true": 1, "false": 0, True: 1, False: 0})
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int8")

    for col in ["cardbin", "lastfourdigits", "euramount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "authtimestamp" in df.columns:
        df["authtimestamp"] = pd.to_datetime(
            df["authtimestamp"], errors="coerce", utc=True
        )
    return df


# ── Drop high-cardinality identifiers ──────────────────────────────────────────

def drop_identifiers(df):
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    return df.drop(columns=cols_to_drop)


# ── Quality report ─────────────────────────────────────────────────────────────

def print_quality_report(df, name="DataFrame"):
    """Print a concise data-quality summary."""
    logger.info("=" * 60)
    logger.info("  Data Quality Report: %s", name)
    logger.info("=" * 60)
    logger.info("Shape: %s", df.shape)

    missing = df.isnull().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    if len(missing) > 0:
        logger.info("Columns with missing values (%d):", len(missing))
        for col, pct in missing.items():
            logger.info("  %-45s %7.2f%%", col, pct * 100)
    else:
        logger.info("No missing values.")

    logger.info("Cardinality (object columns):")
    for col in df.select_dtypes(include="object").columns:
        logger.info("  %-45s %8d unique", col, df[col].nunique())


# ── Full pipeline ──────────────────────────────────────────────────────────────

def run_quality_pipeline(train, test, y_train):
    """Execute all quality steps and return cleaned DataFrames."""
    logger.info("[1/6] Normalising null tokens …")
    train = normalise_nulls(train)
    test = normalise_nulls(test)

    logger.info("[2/6] Deduplicating transactions …")
    train, y_train = deduplicate(train, y_train)
    test = deduplicate(test)
    logger.info("      Train after dedup: %d rows", train.shape[0])
    logger.info("      Test  after dedup: %d rows", test.shape[0])

    logger.info("[3/6] Fixing response code encoding …")
    train = fix_response_codes(train)
    test = fix_response_codes(test)

    logger.info("[4/6] Casting types …")
    train = cast_types(train)
    test = cast_types(test)

    logger.info("[5/6] Dropping identifier columns …")
    # Extract and sanitize transactionid BEFORE dropping it.
    # Removing embedded quotes and commas prevents strict CSV parsers
    # (e.g. DuckDB strict_mode=true) from miscounting columns.
    test_ids = None
    if "transactionid" in test.columns:
        test_ids = (
            test["transactionid"]
            .astype(str)
            .str.replace(r'[",]', '', regex=True)
            .copy()
        )
    train = drop_identifiers(train)
    test = drop_identifiers(test)

    logger.info("[6/6] Printing quality report …")
    print_quality_report(train, "Train (cleaned)")

    return train, test, y_train, test_ids
