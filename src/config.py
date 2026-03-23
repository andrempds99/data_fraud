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
    "log_euramount",
    "card_txn_7d",           # rolling 7-day card velocity (replaces lifetime card_txn_count)
    "card_avg_amount",
    "merchant_txn_count",
    "amount_deviation",
    "hour",
    "dayofweek",
    "day",
    "is_weekend",
    "card_billing_mismatch",
    "geoip_billing_mismatch",
    "domain_freq",
    "cvvused",
    "recurring",
    "initialrecurring",
    "threedsused",
    "success",
    "cardholder_disposabledomain_boolean",
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
