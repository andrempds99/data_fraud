import logging
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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


def add_timestamp_features(df):
    """Extract hour, day-of-week, weekend flag from authtimestamp."""
    df = df.copy()
    if "authtimestamp" in df.columns:
        df["hour"] = df["authtimestamp"].dt.hour
        df["dayofweek"] = df["authtimestamp"].dt.dayofweek
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
        df["day"] = df["authtimestamp"].dt.day
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
                df["card_txn_7d"] = _compute_velocity(
                    df, "saltedhash", "authtimestamp", window_days=7
                )
            df["card_avg_amount"] = df.groupby("saltedhash")["euramount"].transform("mean")
            df["amount_deviation"] = df["euramount"] - df["card_avg_amount"]
        if "merchant" in df.columns:
            df["merchant_txn_count"] = df.groupby("merchant")["merchant"].transform("count")
    else:
        # ── Test / inference mode ────────────────────────────────────────────
        if "saltedhash" in df.columns:
            # Time-windowed velocity: count training transactions for the same
            # card in the 7 days prior to each test transaction's timestamp.
            if "card_ts_7d" in lookup_tables and "authtimestamp" in df.columns:
                window_ns = int(pd.Timedelta(days=7).total_seconds() * 1e9)
                ts_vals = df["authtimestamp"].astype(np.int64).values
                cards = df["saltedhash"].values
                counts = np.zeros(len(df), dtype=np.int32)
                card_ts_map = lookup_tables["card_ts_7d"]
                for i in range(len(df)):
                    ts_list = card_ts_map.get(cards[i], [])
                    if ts_list:
                        t = int(ts_vals[i])
                        lo = np.searchsorted(ts_list, t - window_ns, side='left')
                        hi = np.searchsorted(ts_list, t, side='left')
                        counts[i] = hi - lo
                df["card_txn_7d"] = counts

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

        # Store the last 7 days of training timestamps per card so the test set
        # can resolve time-windowed velocity without leaking test data.
        if "authtimestamp" in train.columns:
            max_ts = train["authtimestamp"].max()
            lookback = max_ts - pd.Timedelta(days=7)
            recent = train.loc[
                train["authtimestamp"].notna() & (train["authtimestamp"] >= lookback),
                ["saltedhash", "authtimestamp"],
            ]
            tables["card_ts_7d"] = {
                card: sorted(grp["authtimestamp"].astype(np.int64).tolist())
                for card, grp in recent.groupby("saltedhash")
            }

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


def add_amount_features(df):
    """Log-transform and other amount-based features."""
    df = df.copy()
    if "euramount" in df.columns:
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

    logger.info("[FE 1/6] Timestamp features …")
    train_fold = add_timestamp_features(train_fold)
    extra_dfs = _apply(add_timestamp_features, extra_dfs)

    logger.info("[FE 2/6] Country mismatch flags …")
    train_fold = add_country_mismatch_flags(train_fold)
    extra_dfs = _apply(add_country_mismatch_flags, extra_dfs)

    logger.info("[FE 3/6] Aggregation / velocity features …")
    lookup_tables = build_lookup_tables(train_fold)
    train_fold = add_aggregation_features(train_fold, lookup_tables=None)
    extra_dfs = _apply(add_aggregation_features, extra_dfs, lookup_tables=lookup_tables)

    logger.info("[FE 4/6] Email domain features …")
    # Always apply domain_freq from lookup_tables (even for train_fold) so
    # that this step is safe to use inside a CV loop without leakage.
    train_fold = add_domain_features(train_fold, lookup_tables=lookup_tables)
    extra_dfs = _apply(add_domain_features, extra_dfs, lookup_tables=lookup_tables)

    logger.info("[FE 5/6] Amount features …")
    train_fold = add_amount_features(train_fold)
    extra_dfs = _apply(add_amount_features, extra_dfs)

    logger.info("[FE 6/6] Dropping consumed raw columns …")
    train_fold = drop_raw_columns(train_fold)
    extra_dfs = _apply(drop_raw_columns, extra_dfs)

    logger.info("  Train features: %d", train_fold.shape[1])

    return train_fold, extra_dfs, lookup_tables
