import logging
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VELOCITY_WINDOWS = [1, 3, 7, 14, 30]


def _compute_velocity(df, group_col, ts_col, window_days=7):
    """Count prior same-entity transactions within a rolling time window (training mode).

    Iterates rows in ascending timestamp order. For each row a binary search
    on the entity's running history counts entries that fall inside
    [t - window, t) — i.e. prior transactions in the look-back window.

    Rows with NaT timestamps receive a count of 0 and do not influence counts
    for other rows.

    Returns a Series aligned to ``df``'s original index.
    """
    window_ns = int(pd.Timedelta(days=window_days).total_seconds() * 1e9)

    result = pd.Series(0, index=df.index, dtype=np.int32)
    valid = df[ts_col].notna()
    if not valid.any():
        return result

    df_valid = df.loc[valid, [ts_col, group_col]].copy()
    sorted_order = df_valid[ts_col].argsort()
    orig_index = df_valid.index[sorted_order.values]
    ts_arr = df_valid[ts_col].iloc[sorted_order].astype(np.int64).values
    grp_arr = df_valid[group_col].iloc[sorted_order].values

    entity_ts: dict = defaultdict(list)   # entity → sorted list of prior timestamps
    out = np.zeros(len(df_valid), dtype=np.int32)

    for pos in range(len(ts_arr)):
        g = grp_arr[pos]
        t = ts_arr[pos]
        prior = entity_ts[g]
        if prior:
            lo = np.searchsorted(prior, t - window_ns, side='left')
            out[pos] = len(prior) - lo
        entity_ts[g].append(int(t))

    result.loc[orig_index] = out
    return result


def _compute_velocity_multi(df, group_col, ts_col, windows=None):
    """Compute velocity for multiple time windows in a single pass.

    Much more efficient than calling ``_compute_velocity`` once per window,
    because the entity history is built only once.

    Returns a dict of {window_days: Series}.
    """
    if windows is None:
        windows = VELOCITY_WINDOWS
    windows_ns = {w: int(pd.Timedelta(days=w).total_seconds() * 1e9) for w in windows}

    results = {w: pd.Series(0, index=df.index, dtype=np.int32) for w in windows}
    valid = df[ts_col].notna()
    if not valid.any():
        return results

    df_valid = df.loc[valid, [ts_col, group_col]].copy()
    sorted_order = df_valid[ts_col].argsort()
    orig_index = df_valid.index[sorted_order.values]
    ts_arr = df_valid[ts_col].iloc[sorted_order].astype(np.int64).values
    grp_arr = df_valid[group_col].iloc[sorted_order].values

    entity_ts: dict = defaultdict(list)
    outs = {w: np.zeros(len(df_valid), dtype=np.int32) for w in windows}

    for pos in range(len(ts_arr)):
        g = grp_arr[pos]
        t = ts_arr[pos]
        prior = entity_ts[g]
        if prior:
            for w in windows:
                lo = np.searchsorted(prior, t - windows_ns[w], side='left')
                outs[w][pos] = len(prior) - lo
        entity_ts[g].append(int(t))

    for w in windows:
        results[w].loc[orig_index] = outs[w]
    return results


def _compute_time_since_last(df, group_col, ts_col):
    """Compute hours since last transaction per entity.

    Returns a Series of float hours (NaN for first-ever transaction).
    """
    df_sorted = df.loc[df[ts_col].notna(), [ts_col, group_col]].copy()
    df_sorted = df_sorted.sort_values(ts_col)
    diff = df_sorted.groupby(group_col)[ts_col].diff()
    result = pd.Series(np.nan, index=df.index, dtype=np.float64)
    result.loc[diff.index] = diff.dt.total_seconds() / 3600.0
    return result


def add_timestamp_features(df):
    """Extract hour, day-of-week, weekend flag, and cyclical encodings from authtimestamp."""
    df = df.copy()
    if "authtimestamp" in df.columns:
        df["hour"] = df["authtimestamp"].dt.hour
        df["dayofweek"] = df["authtimestamp"].dt.dayofweek
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
        df["day"] = df["authtimestamp"].dt.day
        # Cyclical encoding so that hour 23 is close to hour 0
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def add_country_mismatch_flags(df):
    """Flag geographic inconsistencies between card, billing, and IP countries."""
    df = df.copy()
    if "billing_country" in df.columns:
        if "cardcountry" in df.columns:
            df["card_billing_mismatch"] = (
                df["cardcountry"] != df["billing_country"]
            ).astype(int)
        if "geoip_country_code" in df.columns:
            df["geoip_billing_mismatch"] = (
                df["geoip_country_code"] != df["billing_country"]
            ).astype(int)
    return df


def add_aggregation_features(df, lookup_tables=None):
    """Add velocity / aggregation features.

    Training mode (``lookup_tables=None``): compute 7-day rolling card velocity
    and lifetime card / merchant statistics directly from ``df``.

    Test mode (``lookup_tables`` provided): apply pre-built mappings from the
    training set to avoid leaking any test-set information into statistics.
    Card velocity is computed by looking up each test transaction's timestamp
    against the stored last-7-days training timestamps for that card.
    """
    df = df.copy()

    if lookup_tables is None:
        # ── Training mode ────────────────────────────────────────────────────
        if "saltedhash" in df.columns:
            if "authtimestamp" in df.columns:
                velocity_series = _compute_velocity_multi(
                    df, "saltedhash", "authtimestamp", windows=VELOCITY_WINDOWS
                )
                for w, s in velocity_series.items():
                    df[f"card_txn_{w}d"] = s
                df["hours_since_last_txn"] = _compute_time_since_last(
                    df, "saltedhash", "authtimestamp"
                )
            df["card_avg_amount"] = df.groupby("saltedhash")["euramount"].transform("mean")
            df["amount_deviation"] = df["euramount"] - df["card_avg_amount"]
        if "merchant" in df.columns:
            df["merchant_txn_count"] = df.groupby("merchant")["merchant"].transform("count")
        # Card-merchant interaction features
        if "saltedhash" in df.columns and "merchant" in df.columns:
            df["card_merchant_first"] = (
                ~df.duplicated(subset=["saltedhash", "merchant"])
            ).astype(int)
            df["card_unique_merchants"] = df.groupby("saltedhash")["merchant"].transform("nunique")
    else:
        # ── Test / inference mode ────────────────────────────────────────────
        if "saltedhash" in df.columns:
            # Time-windowed velocity for each window
            if "card_ts_all" in lookup_tables and "authtimestamp" in df.columns:
                ts_vals = df["authtimestamp"].astype(np.int64).values
                cards = df["saltedhash"].values
                card_ts_map = lookup_tables["card_ts_all"]
                for w in VELOCITY_WINDOWS:
                    window_ns = int(pd.Timedelta(days=w).total_seconds() * 1e9)
                    counts = np.zeros(len(df), dtype=np.int32)
                    for i in range(len(df)):
                        ts_list = card_ts_map.get(cards[i], [])
                        if ts_list:
                            t = int(ts_vals[i])
                            lo = np.searchsorted(ts_list, t - window_ns, side='left')
                            hi = np.searchsorted(ts_list, t, side='left')
                            counts[i] = hi - lo
                    df[f"card_txn_{w}d"] = counts

            # Hours since last training transaction for this card
            if "card_last_ts" in lookup_tables and "authtimestamp" in df.columns:
                card_last = lookup_tables["card_last_ts"]
                last_ts = df["saltedhash"].map(card_last)
                df["hours_since_last_txn"] = (
                    (df["authtimestamp"] - last_ts).dt.total_seconds() / 3600.0
                )

            if "card_avg_amount" in lookup_tables:
                df["card_avg_amount"] = (
                    df["saltedhash"]
                    .map(lookup_tables["card_avg_amount"])
                    .fillna(lookup_tables["card_avg_amount_default"])
                )
                df["amount_deviation"] = df["euramount"] - df["card_avg_amount"]

        if "merchant" in df.columns and "merchant_txn_count" in lookup_tables:
            df["merchant_txn_count"] = (
                df["merchant"]
                .map(lookup_tables["merchant_txn_count"])
                .fillna(lookup_tables["merchant_txn_count_default"])
            )
        # Card-merchant interaction features (test mode)
        if "saltedhash" in df.columns and "merchant" in df.columns:
            card_merchant_set = lookup_tables.get("card_merchant_pairs", set())
            df["card_merchant_first"] = (
                ~df.apply(
                    lambda r: (r["saltedhash"], r["merchant"]) in card_merchant_set, axis=1
                )
            ).astype(int)
            card_merchant_counts = lookup_tables.get("card_unique_merchants")
            if card_merchant_counts is not None:
                df["card_unique_merchants"] = (
                    df["saltedhash"]
                    .map(card_merchant_counts)
                    .fillna(1)
                )
    return df


def build_lookup_tables(train):
    """Compute aggregation lookup tables from the training set.

    Stores:
    - ``card_ts_7d``: dict of {card -> sorted int64 timestamp list} for the
      last 7 days of training, used to compute time-windowed velocity on unseen
      data without leakage.
    - ``card_avg_amount`` / ``card_avg_amount_default``: lifetime card stats.
    - ``merchant_txn_count`` / ``merchant_txn_count_default``: merchant stats.
    - ``domain_freq`` / ``domain_freq_default``: email domain popularity.
    """
    tables = {}

    if "saltedhash" in train.columns:
        tables["card_avg_amount"] = train.groupby("saltedhash")["euramount"].mean()
        tables["card_avg_amount_default"] = tables["card_avg_amount"].median()

        # Store ALL training timestamps per card (sorted) so test can compute
        # velocity for any window without leaking test data.
        if "authtimestamp" in train.columns:
            max_window = max(VELOCITY_WINDOWS)
            max_ts = train["authtimestamp"].max()
            lookback = max_ts - pd.Timedelta(days=max_window)
            recent = train.loc[
                train["authtimestamp"].notna() & (train["authtimestamp"] >= lookback),
                ["saltedhash", "authtimestamp"],
            ]
            tables["card_ts_all"] = {
                card: sorted(grp["authtimestamp"].astype(np.int64).tolist())
                for card, grp in recent.groupby("saltedhash")
            }
            # Last timestamp per card (for hours_since_last_txn on test)
            tables["card_last_ts"] = train.groupby("saltedhash")["authtimestamp"].max()

        # Card-merchant interaction lookup
        if "merchant" in train.columns:
            tables["card_merchant_pairs"] = set(
                zip(train["saltedhash"], train["merchant"])
            )
            tables["card_unique_merchants"] = train.groupby("saltedhash")["merchant"].nunique()

    if "merchant" in train.columns:
        tables["merchant_txn_count"] = train.groupby("merchant").size()
        tables["merchant_txn_count_default"] = tables["merchant_txn_count"].median()

    if "cardholderdomain" in train.columns:
        tables["domain_freq"] = train["cardholderdomain"].value_counts(normalize=True)
        tables["domain_freq_default"] = 0.0

    return tables


def add_domain_features(df, lookup_tables=None):
    """Add email-domain frequency feature."""
    df = df.copy()
    if "cardholderdomain" not in df.columns:
        return df

    if lookup_tables is None:
        freq = df["cardholderdomain"].value_counts(normalize=True)
        df["domain_freq"] = df["cardholderdomain"].map(freq)
    elif "domain_freq" in lookup_tables:
        df["domain_freq"] = (
            df["cardholderdomain"]
            .map(lookup_tables["domain_freq"])
            .fillna(lookup_tables["domain_freq_default"])
        )
    return df


def add_amount_features(df, lookup_tables=None):
    """Log-transform, outlier capping, and other amount-based features."""
    df = df.copy()
    if "euramount" in df.columns:
        # Winsorize at 99.5th percentile to reduce outlier influence
        if lookup_tables is not None and "euramount_cap" in lookup_tables:
            cap = lookup_tables["euramount_cap"]
        else:
            cap = df["euramount"].quantile(0.995)
        df["euramount_capped"] = df["euramount"].clip(upper=cap)
        df["log_euramount"] = np.log1p(df["euramount"].clip(lower=0))
    return df


def drop_raw_columns(df):
    """Drop columns that have been consumed by feature engineering."""
    to_drop = [
        "authtimestamp",
        "timestamp",
        "saltedhash",
        "merchant",
        "cardholderdomain",
        "cardholder_disposabledomain",
        "responsecode",
        "ddresult",
        "issuingbank",
        "cardcountry",
        "billing_country",
        "geoip_country_code",
        "maxmind_country_code",
        "City",
        "State",
        "customer",
        "eci",
        "mcccode",
        "currency",
        "cardbin",
        "lastfourdigits",
        "merchantcountry",
        "acceptorcountry",
        "transactioncountry",
    ]
    existing = [c for c in to_drop if c in df.columns]
    return df.drop(columns=existing)


def _bin_rare_categories(train_fold, extra_dfs, lookup_tables, min_freq=50):
    """Replace rare category levels (fewer than min_freq occurrences) with '__other__'."""
    from src.config import CATEGORICAL_FEATURES

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in train_fold.columns]
    rare_map = {}
    for col in cat_cols:
        counts = train_fold[col].value_counts()
        frequent = set(counts[counts >= min_freq].index)
        rare_map[col] = frequent
        train_fold[col] = train_fold[col].where(
            train_fold[col].isin(frequent), other="__other__"
        )
        for i, df in enumerate(extra_dfs):
            extra_dfs[i][col] = df[col].where(
                df[col].isin(frequent), other="__other__"
            )
    lookup_tables["rare_category_map"] = rare_map
    return train_fold, extra_dfs, lookup_tables


# ── Full pipeline ──────────────────────────────────────────────────────────────

def run_feature_pipeline(train_fold, extra_dfs=None):
    """Apply all feature engineering to train_fold, then to each df in extra_dfs.

    Lookup tables are built from ``train_fold`` only — this prevents aggregation
    statistics from leaking validation or test rows into training data.

    Args:
        train_fold:  the training split DataFrame (post quality pipeline)
        extra_dfs:   list of additional DataFrames (e.g. [X_val, test]), or None

    Returns:
        train_fold_fe:  engineered training DataFrame
        extra_fes:      list of engineered DataFrames (same order as extra_dfs)
        lookup_tables:  dict of pre-computed mappings (for serialisation / reuse)
    """
    if extra_dfs is None:
        extra_dfs = []

    def _apply(fn, dfs, **kwargs):
        return [fn(df, **kwargs) for df in dfs]

    logger.info("[FE 1/7] Timestamp features (with cyclical encoding) …")
    train_fold = add_timestamp_features(train_fold)
    extra_dfs = _apply(add_timestamp_features, extra_dfs)

    logger.info("[FE 2/7] Country mismatch flags …")
    train_fold = add_country_mismatch_flags(train_fold)
    extra_dfs = _apply(add_country_mismatch_flags, extra_dfs)

    logger.info("[FE 3/7] Aggregation / velocity features (multi-window) …")
    lookup_tables = build_lookup_tables(train_fold)
    train_fold = add_aggregation_features(train_fold, lookup_tables=None)
    extra_dfs = _apply(add_aggregation_features, extra_dfs, lookup_tables=lookup_tables)

    logger.info("[FE 4/7] Email domain features …")
    # Always apply domain_freq from lookup_tables (even for train_fold) so
    # that this step is safe to use inside a CV loop without leakage.
    train_fold = add_domain_features(train_fold, lookup_tables=lookup_tables)
    extra_dfs = _apply(add_domain_features, extra_dfs, lookup_tables=lookup_tables)

    logger.info("[FE 5/7] Amount features (with outlier capping) …")
    # Compute cap from training fold only to avoid leakage
    if "euramount" in train_fold.columns:
        lookup_tables["euramount_cap"] = train_fold["euramount"].quantile(0.995)
    train_fold = add_amount_features(train_fold, lookup_tables=lookup_tables)
    extra_dfs = _apply(add_amount_features, extra_dfs, lookup_tables=lookup_tables)

    logger.info("[FE 6/7] Rare category binning …")
    train_fold, extra_dfs, lookup_tables = _bin_rare_categories(
        train_fold, extra_dfs, lookup_tables, min_freq=50
    )

    logger.info("[FE 7/7] Dropping consumed raw columns …")
    train_fold = drop_raw_columns(train_fold)
    extra_dfs = _apply(drop_raw_columns, extra_dfs)

    logger.info("  Train features: %d", train_fold.shape[1])

    return train_fold, extra_dfs, lookup_tables
