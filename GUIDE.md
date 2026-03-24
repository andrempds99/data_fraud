# Fraud Detection — Data Documentation & Task Guide

> **Objective:** Train a fraud classifier on transaction data, evaluate it rigorously, and produce predictions on a held-out test set.

---

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [Column Dictionary](#2-column-dictionary)
3. [Train vs Test Differences](#3-train-vs-test-differences)
4. [Data Quality Checks](#4-data-quality-checks)
5. [Feature Expansion](#5-feature-expansion)
6. [Controlling Class Imbalance](#6-controlling-class-imbalance)
7. [Training Classifiers](#7-training-classifiers)
8. [Metrics & Plotting](#8-metrics--plotting)
9. [Overfitting Detection & Mitigation](#9-overfitting-detection--mitigation)
10. [Optimising for Different Recalls (30 %–60 %)](#10-optimising-for-different-recalls-30--60-)
11. [Recommended Libraries](#11-recommended-libraries)
12. [End-to-End Workflow Summary](#12-end-to-end-workflow-summary)

---

## 1. Dataset Overview

| Property | Train | Test |
|---|---|---|
| File | `train_data.csv` | `test_data.csv` |
| Rows | **157 306** | **110 998** |
| Columns | **48** | **44** |
| Target column | `cb_fraudflag` | *(not present — must be predicted)* |
| Date range (sample) | starts from 2020-02-29 | starts from 2020-04-30 |
| File encoding | UTF-8 (contains some garbled characters) | UTF-8 |

The training set contains **4 extra columns** that relate to fraud/chargeback labels. These columns are **absent from the test set** and must **never be used as features** (they are labels or label-proxies and would cause data leakage).

### Target Variable

**`cb_fraudflag`** — a numeric flag (observed values: `0.0`, `1.0`) indicating whether a chargeback/fraud event was associated with the transaction.

- `0.0` → legitimate transaction
- `1.0` → fraudulent / chargebacked transaction

> **Expect severe class imbalance:** fraud rates in payments data are typically 0.5 %–3 %. Verify the actual ratio early (see Section 4).

---

## 2. Column Dictionary

All 48 train columns are listed below, grouped by domain. The 4 columns marked with ⚠️ exist **only in `train_data.csv`** and must not be used as features.

### 2.1 Transaction Identifiers & Metadata

| # | Column | Type | Description | Example |
|---|---|---|---|---|
| 0 | `currency` | int/str | ISO 4217 numeric currency code | `643` (RUB), `840` (USD), `978` (EUR) |
| 1 | `transactionid` | str | Unique transaction identifier | `28302-07053-30612` |
| 2 | `merchant` | str | Merchant ID and name (hyphenated) | `9782-illicitpassion` |
| 3 | `merchanturl` | str | Merchant website URL | `illicitpassion.com` |
| 4 | `submerchant` | str | Sub-merchant identifier | `1cillicitpassion.com` |
| 5 | `transactiontype` | str | Type of transaction | `auth`, `auth_capture`, `capture`, `refund` |
| 6 | `responsecode` | str | Gateway/processor response code | `none`, `paraminvalid:cccvc`, garbled text |
| 7 | `decline_type` | str | Category of decline (if any) | `PROCESSING`, `ANTIFRAUD`, `""` |
| 8 | `ddresult` | str | Due-diligence / risk-scoring result label | `subscription`, `product`, `deposit...` |
| 9 | `currencyname` | str | ISO 4217 alphabetic currency code | `RUB`, `USD`, `EUR` |
| 24 | `approval_code` | str | Authorisation approval code | `""`, alphanumeric codes |
| 34 | `channel` | str | Transaction channel | `ecom` |
| 35 | `terminaltype` | str | Terminal category | `cat6` |
| 38 | `success` | bool/str | Whether the transaction was successfully processed | `true`, `false` |
| 39 | `euramount` | float | Transaction amount converted to EUR | `430.0`, `106.0`, `111.66` |

### 2.2 Card & Issuer Information

| # | Column | Type | Description | Example |
|---|---|---|---|---|
| 10 | `cardbrand` | str | Card network brand | `visa`, `master`, `amex` |
| 11 | `cardtype` | str | Card type | `debit`, `credit` |
| 12 | `brandcardtype` | str | Card product tier | `standard`, `world`, `classic`, `electron`, `prepaid` |
| 13 | `issuingbank` | str | Name of the card-issuing bank | `sberbankofrussia`, `tinkoffbank` |
| 14 | `cardcountry` | str | Country where the card was issued (full name) | `russianfederation`, `france`, `italy` |
| 15 | `saltedhash` | str | SHA-256 salted hash of the card PAN | 64-char hex string |
| 40 | `cardbin` | int/str | First 6 digits of the card (BIN) | `546955`, `497831`, `none` |
| 41 | `lastfourdigits` | int/str | Last 4 digits of the card | `8496`, `0496`, `none` |

### 2.3 Cardholder & Billing Address

| # | Column | Type | Description | Example |
|---|---|---|---|---|
| 16 | `billing_country` | str | Billing address country (ISO 2-letter) | `RU`, `FR`, `IT` |
| 17 | `City` | str | Billing city | `ECCICA-SUARELLA`, `Varese`, `n/a` |
| 18 | `State` | str | Billing state/province | `N/A`, `NA`, `""` |
| 28 | `customer` | str | Customer/processor identifier | `ikajo` |

### 2.4 Geography & IP

| # | Column | Type | Description | Example |
|---|---|---|---|---|
| 19 | `transactionip` | str | IP address of the transaction | `176.59.4.5` |
| 20 | `geoip_country_code` | str | Country from GeoIP lookup (ISO 2-letter) | `RU`, `FR`, `""` |
| 21 | `maxmind_country_code` | str | Country from MaxMind lookup (ISO 2-letter) | `RU`, `IT`, `""` |
| 42 | `merchantcountry` | str | Merchant's registered country | `none`, ISO codes |
| 43 | `acceptorcountry` | str | Acquirer/acceptor country | `none`, ISO codes |
| 44 | `transactioncountry` | str | Country where the transaction was processed | `none`, ISO codes |

### 2.5 Authentication & Security

| # | Column | Type | Description | Example |
|---|---|---|---|---|
| 22 | `cvvused` | bool/str | Whether CVV was provided | `true`, `false` |
| 23 | `recurring` | bool/str | Whether this is a recurring payment | `true`, `false` |
| 27 | `eci` | str | Electronic Commerce Indicator (3DS result) | `none`, ECI codes |
| 28 | `mcccode` | str | Merchant Category Code | `none`, numeric codes |
| 36 | `initialrecurring` | bool/str | Whether this is the first transaction in a recurring series | `true`, `false` |
| 37 | `threedsused` | bool/str | Whether 3-D Secure was used | `true`, `false` |

### 2.6 Email / Domain

| # | Column | Type | Description | Example |
|---|---|---|---|---|
| 45 | `cardholderdomain` | str | Email domain of the cardholder | `gmail.com`, `mail.ru` |
| 46 | `cardholder_disposabledomain` | str | Disposable email domain (if applicable) | `none`, domain string |
| 47 | `cardholder_disposabledomain_boolean` | bool/str | Whether the email domain is disposable | `true`, `false` |

### 2.7 Timestamps

| # | Column | Type | Description | Example |
|---|---|---|---|---|
| 29 | `authtimestamp` | str (ISO 8601) | Full authorisation timestamp | `2020-04-30T23:50:00.000Z` |
| 30 | `timestamp` | int (epoch) | Unix epoch timestamp (seconds) | `1588290600` |

### 2.8 ⚠️ Label / Chargeback Columns (Train Only — DO NOT USE AS FEATURES)

| # | Column | Type | Description | Example |
|---|---|---|---|---|
| 25 | `chargeback_bank_date` | str/date | Date the bank raised the chargeback | `""`, date string |
| 26 | `chargeback_reason_code` | str | Reason code for the chargeback | `""`, reason code |
| 32 | `fraudimportdate` | int (epoch) | Epoch timestamp when fraud was reported/imported | `1514764800` |
| 33 | `cb_fraudflag` | float | **TARGET** — `1.0` = fraud, `0.0` = legitimate | `0.0`, `1.0` |

> **Critical:** Using any of these four columns as features will leak the label into the model and produce unrealistically high metrics. They must be dropped before any feature engineering or model training.

---

## 3. Train vs Test Differences

| Aspect | `train_data.csv` | `test_data.csv` |
|---|---|---|
| Rows | 157 306 | 110 998 |
| Columns | 48 | 44 |
| `cb_fraudflag` (target) | ✅ Present | ❌ Absent |
| `chargeback_bank_date` | ✅ Present | ❌ Absent |
| `chargeback_reason_code` | ✅ Present | ❌ Absent |
| `fraudimportdate` | ✅ Present | ❌ Absent |
| Date range | Appears to start earlier (~Feb 2020) | Appears to start later (~Apr 2020) |

**Implication:** The test set is a temporal hold-out. Your model must generalise to future, unseen transactions.

---

## 4. Data Quality Checks

Before modelling, audit the data for issues that could silently degrade model performance.

### 4.1 Loading the Data

```python
import pandas as pd
import numpy as np

train = pd.read_csv("train_data.csv", encoding="utf-8", encoding_errors="replace")
test  = pd.read_csv("test_data.csv",  encoding="utf-8", encoding_errors="replace")

# Separate target and drop label columns immediately
TARGET = "cb_fraudflag"
LABEL_COLS = ["cb_fraudflag", "chargeback_bank_date", "chargeback_reason_code", "fraudimportdate"]

y_train = train[TARGET].copy()
train.drop(columns=LABEL_COLS, inplace=True)

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Fraud rate: {y_train.mean():.4%}")
```

### 4.2 Missing / Null Values

This dataset uses **multiple representations** for missing data. Normalise them before analysis.

```python
# Unify null representations
NULL_TOKENS = {"none", "n/a", "na", "nan", "null", ""}

def normalise_nulls(df):
    """Replace known null tokens with np.nan across all object columns."""
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip().str.lower()
        df[col] = df[col].replace(NULL_TOKENS, np.nan)
    return df

train = normalise_nulls(train)
test  = normalise_nulls(test)

# Missing value report
missing = train.isnull().mean().sort_values(ascending=False)
print(missing[missing > 0])
```

### 4.3 Duplicate Transactions

A single purchase can generate multiple rows (e.g. `auth` followed by `capture`). Decide on a deduplication strategy.

```python
# Check duplicates on transactionid
dup_counts = train.groupby("transactionid")["transactiontype"].apply(list)
multi_type = dup_counts[dup_counts.apply(len) > 1]
print(f"Transaction IDs with multiple rows: {len(multi_type)}")
print(multi_type.head(10))

# Common strategy: keep only the final-state row per transaction
# e.g. prefer 'capture' or 'auth_capture' over plain 'auth'
TYPE_PRIORITY = {"refund": 0, "auth_capture": 1, "capture": 2, "auth": 3}
train["_type_priority"] = train["transactiontype"].map(TYPE_PRIORITY).fillna(99)
train.sort_values("_type_priority", inplace=True)
train.drop_duplicates(subset="transactionid", keep="first", inplace=True)
train.drop(columns="_type_priority", inplace=True)
```

### 4.4 Encoding Issues

The `responsecode` column contains garbled text (e.g. `limitгјberschritten` instead of `limitüberschritten`). This is a character-encoding artefact.

```python
# Inspect unique response codes
print(train["responsecode"].value_counts().head(20))

# Option 1: Group rare / garbled codes into an "other" bucket
# Option 2: Fix known encoding issues with a mapping dict
RESPONSE_FIXES = {
    "limitгјberschritten,doch-funktionmг¶glich": "limit_exceeded_possible",
    # Add more as discovered
}
train["responsecode"] = train["responsecode"].replace(RESPONSE_FIXES)
```

### 4.5 Data Type Consistency

```python
# Boolean columns stored as strings
BOOL_COLS = ["cvvused", "recurring", "initialrecurring", "threedsused",
             "success", "cardholder_disposabledomain_boolean"]

for col in BOOL_COLS:
    for df in [train, test]:
        df[col] = df[col].map({"true": 1, "false": 0}).astype("Int8")

# Numeric columns that may contain "none"
for col in ["cardbin", "lastfourdigits", "euramount"]:
    for df in [train, test]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Timestamp parsing
for df in [train, test]:
    df["authtimestamp"] = pd.to_datetime(df["authtimestamp"], errors="coerce", utc=True)
```

### 4.6 Cardinality Check

High-cardinality columns may need special encoding or should be dropped.

```python
for col in train.select_dtypes(include="object").columns:
    nuniq = train[col].nunique()
    print(f"{col:40s} {nuniq:>8,} unique")
```

> **Watch out for:** `saltedhash` (unique per card, extremely high cardinality — likely needs to be dropped or used only for aggregation), `transactionid` (identifier — drop), `transactionip` (high cardinality — consider aggregation or geolocation features instead).

---

## 5. Feature Expansion

Feature engineering is where domain knowledge adds the most value in fraud detection.

### 5.1 Timestamp Features

```python
train["hour"]       = train["authtimestamp"].dt.hour
train["dayofweek"]  = train["authtimestamp"].dt.dayofweek   # 0=Mon, 6=Sun
train["is_weekend"] = train["dayofweek"].isin([5, 6]).astype(int)
train["day"]        = train["authtimestamp"].dt.day
```

### 5.2 Country Mismatch Flags

Geographic inconsistencies are strong fraud signals.

```python
# Card country vs billing country
train["card_billing_mismatch"] = (
    train["cardcountry"] != train["billing_country"]
).astype(int)

# GeoIP vs billing country
train["geoip_billing_mismatch"] = (
    train["geoip_country_code"] != train["billing_country"]
).astype(int)

# GeoIP vs card country (need to normalise — cardcountry is full name, geoip is ISO code)
# Consider creating a mapping for cardcountry → ISO 2-letter code first
```

### 5.3 Aggregation / Velocity Features

Transaction patterns over time are highly predictive. Build these on the **training set only** (to avoid leakage), then apply the same mapping to test.

```python
# Number of transactions per cardholder (saltedhash)
card_txn_count = train.groupby("saltedhash")["transactionid"].transform("count")
train["card_txn_count"] = card_txn_count

# Average amount per cardholder
train["card_avg_amount"] = train.groupby("saltedhash")["euramount"].transform("mean")

# Transaction count per merchant
train["merchant_txn_count"] = train.groupby("merchant")["transactionid"].transform("count")

# Deviation from cardholder's average amount
train["amount_deviation"] = train["euramount"] - train["card_avg_amount"]
```

> **Tip:** For the test set, compute lookup tables from train and use `.map()`. For unknown keys (new cards/merchants), fill with a global default (e.g. median).

### 5.4 Email Domain Risk Features

```python
# Already have cardholder_disposabledomain_boolean
# Additionally, compute email domain frequency
domain_freq = train["cardholderdomain"].value_counts(normalize=True)
train["domain_freq"] = train["cardholderdomain"].map(domain_freq)

# Fraud rate per email domain (use with care — potential leakage if not done via CV)
domain_fraud_rate = train.groupby("cardholderdomain")[TARGET].mean()  # compute on train only
# ... use only inside a cross-validation loop to avoid leaking test-fold labels
```

### 5.5 Amount-Based Features

```python
# Log-transform to reduce skew
train["log_euramount"] = np.log1p(train["euramount"])

# Bin amounts
train["amount_bucket"] = pd.cut(
    train["euramount"],
    bins=[0, 10, 50, 100, 250, 500, 1000, 5000, np.inf],
    labels=["0-10", "10-50", "50-100", "100-250", "250-500", "500-1k", "1k-5k", "5k+"]
)
```

### 5.6 Encoding Categorical Features

```python
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# Low-cardinality categoricals: one-hot or ordinal encoding
LOW_CARD = ["cardbrand", "cardtype", "transactiontype", "channel", "terminaltype"]
train_encoded = pd.get_dummies(train, columns=LOW_CARD, drop_first=True)

# Medium-cardinality: frequency / target encoding
# Frequency encoding example
for col in ["issuingbank", "brandcardtype", "merchanturl"]:
    freq = train[col].value_counts(normalize=True)
    train[col + "_freq"] = train[col].map(freq)
```

> **Important:** Apply the **same** encoding mappings (built on train) to the test set. Never fit encoders on test data.

---

## 6. Controlling Class Imbalance

Fraud datasets are heavily imbalanced. If the fraud rate is ~1 %, a model that always predicts "legitimate" achieves 99 % accuracy but catches zero fraud.

### 6.1 Measure the Imbalance

```python
print(y_train.value_counts())
print(f"Fraud rate: {y_train.mean():.4%}")
print(f"Imbalance ratio: 1:{int((1 - y_train.mean()) / y_train.mean())}")
```

### 6.2 Strategies

#### A. Class Weights (Simplest — Start Here)

Most sklearn classifiers accept `class_weight="balanced"`:

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(class_weight="balanced", random_state=42)
```

For XGBoost / LightGBM use `scale_pos_weight`:

```python
import xgboost as xgb

ratio = (y_train == 0).sum() / (y_train == 1).sum()
model = xgb.XGBClassifier(scale_pos_weight=ratio, random_state=42)
```

#### B. Resampling with `imbalanced-learn`

```python
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

# SMOTE oversampling
smote = SMOTE(sampling_strategy=0.5, random_state=42)  # 0.5 = minority becomes 50% of majority
X_res, y_res = smote.fit_resample(X_train, y_train)

# Combined: oversample minority + undersample majority
pipeline_resample = ImbPipeline([
    ("smote", SMOTE(sampling_strategy=0.3, random_state=42)),
    ("undersample", RandomUnderSampler(sampling_strategy=0.5, random_state=42)),
])
X_res, y_res = pipeline_resample.fit_resample(X_train, y_train)
```

#### C. ADASYN (Adaptive Synthetic Sampling)

```python
adasyn = ADASYN(sampling_strategy=0.5, random_state=42)
X_res, y_res = adasyn.fit_resample(X_train, y_train)
```

> **Best Practice:** Always **evaluate on the original (unsampled) validation/test distribution**. Only resample the training fold inside cross-validation.

---

## 7. Training Classifiers

### 7.1 Train / Validation Split

```python
from sklearn.model_selection import train_test_split

X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)
print(f"Train: {X_train_split.shape}, Val: {X_val.shape}")
print(f"Val fraud rate: {y_val.mean():.4%}")
```

> **Alternative:** Use **time-based split** since this is temporal data. Sort by `timestamp` and use the last N% as validation. This more closely mirrors production conditions.

```python
train_sorted = train.sort_values("timestamp")
split_idx = int(len(train_sorted) * 0.8)
X_train_split = train_sorted.iloc[:split_idx]
X_val = train_sorted.iloc[split_idx:]
```

### 7.2 Preprocessing Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

numeric_features = ["euramount", "log_euramount", "card_txn_count",
                    "card_avg_amount", "amount_deviation", "hour", "dayofweek"]
categorical_features = ["cardbrand", "cardtype", "transactiontype", "channel"]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])
```

### 7.3 Baseline Models

#### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

lr_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
])
lr_pipe.fit(X_train_split, y_train_split)
```

#### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        max_depth=15,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )),
])
rf_pipe.fit(X_train_split, y_train_split)
```

#### XGBoost

```python
import xgboost as xgb

ratio = (y_train_split == 0).sum() / (y_train_split == 1).sum()

xgb_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=ratio,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )),
])
# For early stopping with XGBoost in a pipeline, fit with eval_set
xgb_pipe.fit(X_train_split, y_train_split)
```

#### LightGBM

```python
import lightgbm as lgb

lgb_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        is_unbalance=True,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )),
])
lgb_pipe.fit(X_train_split, y_train_split)
```

---

## 8. Metrics & Plotting

For fraud detection, **accuracy is misleading**. Focus on **precision-recall** metrics.

### 8.1 Core Metrics

```python
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, f1_score
)

y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]

print(classification_report(y_val, y_pred, target_names=["Legit", "Fraud"]))
print(f"ROC-AUC:  {roc_auc_score(y_val, y_prob):.4f}")
print(f"PR-AUC:   {average_precision_score(y_val, y_prob):.4f}")
```

### 8.2 Confusion Matrix

```python
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_val, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
            xticklabels=["Legit", "Fraud"],
            yticklabels=["Legit", "Fraud"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
```

### 8.3 Precision-Recall Curve

```python
precision, recall, thresholds = precision_recall_curve(y_val, y_prob)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(recall, precision, lw=2)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title(f"Precision-Recall Curve (AP = {average_precision_score(y_val, y_prob):.3f})")
ax.axhline(y=y_val.mean(), color="grey", linestyle="--", label="Baseline (fraud rate)")
ax.legend()
plt.tight_layout()
plt.savefig("precision_recall_curve.png", dpi=150)
plt.show()
```

### 8.4 ROC Curve

```python
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_val, y_prob)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_score(y_val, y_prob):.3f}")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate (Recall)")
ax.set_title("ROC Curve")
ax.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.show()
```

### 8.5 Feature Importance (Tree Models)

```python
# For tree-based models in a pipeline
classifier = model.named_steps["classifier"]
importances = classifier.feature_importances_

# Get feature names from the preprocessor
feature_names = model.named_steps["preprocessor"].get_feature_names_out()

feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
feat_imp.head(20).plot(kind="barh", figsize=(8, 6), title="Top 20 Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()
```

---

## 9. Overfitting Detection & Mitigation

### 9.1 Detecting Overfitting

Overfitting manifests as a **large gap between training and validation performance**.

```python
# Compare train vs validation metrics
for name, X, y in [("Train", X_train_split, y_train_split), ("Val", X_val, y_val)]:
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    print(f"{name:6s}  ROC-AUC={roc_auc_score(y, y_prob):.4f}  "
          f"PR-AUC={average_precision_score(y, y_prob):.4f}  "
          f"F1={f1_score(y, y_pred):.4f}")
```

> **Red flag:** Train AUC ≈ 1.00 but Val AUC ≈ 0.75 → severe overfitting.

### 9.2 Learning Curves

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring="average_precision",
    n_jobs=-1, random_state=42
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(train_sizes, train_scores.mean(axis=1), label="Train")
ax.plot(train_sizes, val_scores.mean(axis=1), label="Validation")
ax.fill_between(train_sizes,
                train_scores.mean(axis=1) - train_scores.std(axis=1),
                train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
ax.fill_between(train_sizes,
                val_scores.mean(axis=1) - val_scores.std(axis=1),
                val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
ax.set_xlabel("Training Set Size")
ax.set_ylabel("Average Precision")
ax.set_title("Learning Curves")
ax.legend()
plt.tight_layout()
plt.savefig("learning_curves.png", dpi=150)
plt.show()
```

### 9.3 Mitigation Strategies

| Strategy | How | Applicable to |
|---|---|---|
| **Regularisation** | Increase L1/L2 penalty (C parameter in LogReg) | Logistic Regression |
| **Tree depth limits** | Reduce `max_depth`, increase `min_samples_leaf` | RF, XGBoost, LightGBM |
| **Early stopping** | Stop training when validation metric stops improving | XGBoost, LightGBM |
| **Subsampling** | Reduce `subsample` and `colsample_bytree` | XGBoost, LightGBM |
| **Feature selection** | Remove noisy / leaky features | All |
| **Cross-validation** | Use stratified k-fold instead of a single split | All |

#### Early Stopping Example (XGBoost)

```python
import xgboost as xgb

# Preprocess first
X_train_proc = preprocessor.fit_transform(X_train_split)
X_val_proc   = preprocessor.transform(X_val)

xgb_model = xgb.XGBClassifier(
    n_estimators=1000,      # high ceiling
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=ratio,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="aucpr",
    early_stopping_rounds=30,
    random_state=42,
)
xgb_model.fit(
    X_train_proc, y_train_split,
    eval_set=[(X_val_proc, y_val)],
    verbose=50,
)
print(f"Best iteration: {xgb_model.best_iteration}")
```

#### Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv,
                         scoring="average_precision", n_jobs=-1)
print(f"CV PR-AUC: {scores.mean():.4f} ± {scores.std():.4f}")
```

---

## 10. Optimising for Different Recalls (30 %–60 %)

The task asks to optimise for recall between 30 % and 60 %. In fraud detection, recall = the proportion of actual frauds that the model catches. Higher recall catches more fraud but also increases false positives (legitimate transactions flagged as fraud).

### 10.1 Understanding the Trade-off

- **Recall 30 %** → catches 30 % of fraud, fewer false alerts, higher precision
- **Recall 60 %** → catches 60 % of fraud, more false alerts, lower precision

The right operating point depends on the **cost of missing fraud vs the cost of blocking a legitimate customer**.

### 10.2 Finding Thresholds for Target Recalls

By default, `predict()` uses a threshold of 0.5. You can tune this threshold to achieve any desired recall.

```python
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score

y_prob = model.predict_proba(X_val)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)

# Find threshold for each target recall
target_recalls = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

print(f"{'Target Recall':>15s} {'Threshold':>10s} {'Actual Recall':>15s} "
      f"{'Precision':>10s} {'F1':>8s}")
print("-" * 65)

for target in target_recalls:
    # Find the highest threshold that achieves at least target recall
    valid_idx = np.where(recalls >= target)[0]
    if len(valid_idx) == 0:
        continue
    # Among those, pick the one with highest precision (= highest threshold)
    best_idx = valid_idx[np.argmax(precisions[valid_idx])]
    thr = thresholds[min(best_idx, len(thresholds) - 1)]

    y_pred_custom = (y_prob >= thr).astype(int)
    actual_recall = recall_score(y_val, y_pred_custom)
    actual_prec   = precision_score(y_val, y_pred_custom)
    actual_f1     = f1_score(y_val, y_pred_custom)

    print(f"{target:>15.0%} {thr:>10.4f} {actual_recall:>15.2%} "
          f"{actual_prec:>10.2%} {actual_f1:>8.4f}")
```

### 10.3 Visualising the Operating Points

```python
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(recalls, precisions[:-1] if len(precisions) > len(recalls) else precisions, lw=2)

# Mark target recall points
for target in target_recalls:
    valid_idx = np.where(recalls >= target)[0]
    if len(valid_idx) == 0:
        continue
    best_idx = valid_idx[np.argmax(precisions[valid_idx])]
    ax.plot(recalls[best_idx], precisions[best_idx], "ro", markersize=8)
    ax.annotate(f"R={target:.0%}", (recalls[best_idx], precisions[best_idx]),
                textcoords="offset points", xytext=(10, 5), fontsize=9)

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Operating Points at Target Recalls")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig("recall_operating_points.png", dpi=150)
plt.show()
```

### 10.4 Generating Test Predictions at a Chosen Threshold

```python
# Choose your operating threshold (e.g. for ~50% recall)
CHOSEN_THRESHOLD = 0.25  # adjust based on the table above

# Preprocess test set with the same pipeline
X_test_proc = preprocessor.transform(test)
test_probs  = model.predict_proba(X_test_proc)[:, 1]
test_preds  = (test_probs >= CHOSEN_THRESHOLD).astype(int)

# Save predictions
output = pd.DataFrame({
    "transactionid": test["transactionid"],
    "fraud_probability": test_probs,
    "fraud_prediction": test_preds,
})
output.to_csv("test_predictions.csv", index=False)
print(f"Predictions saved: {output['fraud_prediction'].value_counts().to_dict()}")
```

---

## 11. Recommended Libraries

```
pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm matplotlib seaborn
```

| Library | Version | Purpose |
|---|---|---|
| `pandas` | ≥ 1.5 | Data loading, manipulation |
| `numpy` | ≥ 1.24 | Numeric operations |
| `scikit-learn` | ≥ 1.3 | Pipelines, models, metrics, cross-validation |
| `imbalanced-learn` | ≥ 0.11 | SMOTE, ADASYN, resampling pipelines |
| `xgboost` | ≥ 2.0 | Gradient boosting classifier |
| `lightgbm` | ≥ 4.0 | Gradient boosting classifier |
| `matplotlib` | ≥ 3.7 | Plotting |
| `seaborn` | ≥ 0.12 | Plotting |

---

## 12. End-to-End Workflow Summary

```text
┌──────────────────────────────────────────────────────────────────┐
│  1. LOAD DATA                                                    │
│     • Read CSVs (UTF-8)                                          │
│     • Separate target (cb_fraudflag), drop label cols            │
│     • Print shapes, fraud rate                                   │
├──────────────────────────────────────────────────────────────────┤
│  2. DATA QUALITY                                                 │
│     • Normalise null tokens (none, n/a, "")                      │
│     • Deduplicate by transactionid                               │
│     • Fix encoding artefacts                                     │
│     • Cast booleans/numerics, parse timestamps                   │
│     • Review cardinality, drop identifiers                       │
├──────────────────────────────────────────────────────────────────┤
│  3. FEATURE ENGINEERING                                          │
│     • Time features (hour, weekday, weekend)                     │
│     • Country mismatch flags                                     │
│     • Velocity / aggregation features                            │
│     • Email domain features                                      │
│     • Amount transforms (log, bins)                              │
│     • Encode categoricals (one-hot, frequency)                   │
├──────────────────────────────────────────────────────────────────┤
│  4. SPLIT DATA                                                   │
│     • Stratified or time-based train/validation split            │
├──────────────────────────────────────────────────────────────────┤
│  5. HANDLE IMBALANCE                                             │
│     • class_weight / scale_pos_weight                            │
│     • Optionally: SMOTE / ADASYN on train fold only              │
├──────────────────────────────────────────────────────────────────┤
│  6. TRAIN MODELS                                                 │
│     • Logistic Regression (baseline)                             │
│     • Random Forest                                              │
│     • XGBoost / LightGBM (with early stopping)                  │
├──────────────────────────────────────────────────────────────────┤
│  7. EVALUATE                                                     │
│     • PR-AUC, ROC-AUC, F1, confusion matrix                     │
│     • Compare train vs val metrics (overfit check)               │
│     • Learning curves, cross-validation                          │
├──────────────────────────────────────────────────────────────────┤
│  8. TUNE RECALL                                                  │
│     • Sweep thresholds on validation set                         │
│     • Report precision/F1 at recall = 30%, 35%, …, 60%          │
│     • Choose operating point based on business constraints       │
├──────────────────────────────────────────────────────────────────┤
│  9. PREDICT ON TEST                                              │
│     • Apply same pipeline to test_data.csv                       │
│     • Generate predictions at chosen threshold                   │
│     • Save to test_predictions.csv                               │
└──────────────────────────────────────────────────────────────────┘
```

---

*Document generated for the Fraudio data science take-home assessment.*
