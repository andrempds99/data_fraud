import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV = os.path.join(BASE_DIR, "train_data.csv")
TEST_CSV = os.path.join(BASE_DIR, "test_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# ── Target ─────────────────────────────────────────────────────────────────────
TARGET = "cb_fraudflag"
LABEL_COLS = [
    "cb_fraudflag",
    "chargeback_bank_date",
    "chargeback_reason_code",
    "fraudimportdate",
]

# ── Columns to drop (identifiers / high-cardinality with no signal) ───────────
DROP_COLS = [
    "transactionid",
    # saltedhash is NOT dropped here — features.py needs it for velocity
    # aggregation; drop_raw_columns() in features.py removes it after FE.
    "transactionip",
    "approval_code",
    "submerchant",
    "merchanturl",
]

# ── Boolean columns stored as strings ─────────────────────────────────────────
BOOL_COLS = [
    "cvvused",
    "recurring",
    "initialrecurring",
    "threedsused",
    "success",
    "cardholder_disposabledomain_boolean",
]

# ── Null tokens present in the data ───────────────────────────────────────────
NULL_TOKENS = {"none", "n/a", "na", "nan", "null", ""}

# ── Deduplication priority (lower = keep) ─────────────────────────────────────
TYPE_PRIORITY = {"refund": 0, "auth_capture": 1, "capture": 2, "auth": 3}

# ── Response-code encoding fixes ──────────────────────────────────────────────
RESPONSE_FIXES = {
    "limitгјberschritten,doch-funktionmг¶glich": "limit_exceeded_possible",
}

# ── Feature groups for the preprocessor ───────────────────────────────────────
NUMERIC_FEATURES = [
    "euramount",
    "euramount_capped",
    "log_euramount",
    "card_txn_1d",           # rolling 1-day card velocity
    "card_txn_3d",           # rolling 3-day card velocity
    "card_txn_7d",           # rolling 7-day card velocity
    "card_txn_14d",          # rolling 14-day card velocity
    "card_txn_30d",          # rolling 30-day card velocity
    "hours_since_last_txn",  # hours since last transaction for this card
    "card_avg_amount",
    "merchant_txn_count",
    "amount_deviation",
    "hour",
    "dayofweek",
    "day",
    "is_weekend",
    "hour_sin",              # cyclical hour encoding (sin)
    "hour_cos",              # cyclical hour encoding (cos)
    "dow_sin",               # cyclical day-of-week encoding (sin)
    "dow_cos",               # cyclical day-of-week encoding (cos)
    "card_billing_mismatch",
    "geoip_billing_mismatch",
    "domain_freq",
    "cvvused",
    "recurring",
    "initialrecurring",
    "threedsused",
    "success",
    "cardholder_disposabledomain_boolean",
    "card_merchant_first",   # first time this card uses this merchant
    "card_unique_merchants", # number of distinct merchants for this card
    "amount_velocity_7d",    # euramount × card_txn_7d (compound risk signal)
    "amount_ratio",          # euramount / card_avg_amount (relative size)
    "velocity_accel",        # card_txn_1d - card_txn_7d/7 (burst detection)
    "merchant_te",           # target-encoded merchant fraud propensity
    "issuingbank_te",        # target-encoded issuing bank fraud propensity
]

CATEGORICAL_FEATURES = [
    "cardbrand",
    "cardtype",
    "transactiontype",
    "channel",
    "terminaltype",
    "currencyname",
    "brandcardtype",
    "decline_type",
]

# ── Recall targets ────────────────────────────────────────────────────────────
TARGET_RECALLS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

# ── Random seed ───────────────────────────────────────────────────────────────
SEED = 42
