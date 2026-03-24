# Fraud Detection Pipeline

## Problem

Build a classifier to detect fraudulent transactions in payment data. 

## Approach

### 1. Data Quality

Identified and addressed several issues during exploration:

- **Null sentinels**: Values like `"none"`, `"n/a"`, `"null"` normalised to `NaN`
- **Duplicates**: ~28% of rows are duplicate `transactionid` entries (auth + capture pairs). Fraud-labelled rows are preserved during deduplication
- **100% null columns**: 4 columns (`merchantcountry`, `acceptorcountry`, `transactioncountry`, `eci`) are entirely empty — dropped automatically
- **Encoding issues**: Corrupted response code strings fixed
- **High-cardinality identifiers**: `transactionid`, `transactionip`, `approval_code` etc. dropped (no predictive value, leakage risk)

### 2. Feature Engineering

39 features (31 numeric + 8 categorical) across 6 categories:

| Category | Features | Rationale |
|----------|----------|-----------|
| **Temporal** | Hour, day-of-week, weekend flag, cyclical sin/cos | Fraud patterns vary by time of day and weekday |
| **Velocity** | Card txn count in 1/3/7/14/30-day windows | Rapid card use signals a compromised card |
| **Behavioural** | Time since last txn, first-time merchant, unique merchant count | Unusual patterns for the cardholder |
| **Geographic** | Card-billing mismatch, GeoIP-billing mismatch | Cross-border fraud indicator |
| **Amount** | Log-transform, 99.5th-percentile cap, deviation from card average | Outlier transactions are suspicious |
| **Domain** | Cardholder email domain frequency | Disposable or rare domains correlate with fraud |

**Leakage prevention**: All aggregation statistics (velocity, averages) are computed from the training fold only, stored as lookup tables, and applied to validation/test sets without information leakage.

### 3. Class Imbalance

The pipeline compares two strategies and automatically picks the better one:

| Strategy | How it works | Trade-off |
|----------|-------------|-----------|
| **Class weights** | Penalises misclassifying fraud more heavily during loss computation | Simpler; preserves real data distribution |
| **SMOTE** | Synthesises minority-class examples via k-NN interpolation | Can help models learn rarer decision boundaries |

Both are run on the best-performing model type. Whichever yields higher PR-AUC on the validation set is selected. In practice, class weights tend to outperform SMOTE on this dataset because the fraud count is extremely low and SMOTE's synthetic samples may not overlap with the true fraud distribution.

### 4. Classifiers

| Model | Why included |
|-------|-------------|
| Logistic Regression | Interpretable linear baseline |
| Random Forest | Non-linear, handles mixed features well |
| XGBoost | Gradient boosting — strong fraud detection baseline |
| LightGBM | Fast gradient boosting alternative |
| Stacking Ensemble | Combines base model predictions via logistic meta-learner |

All boosting models use **early stopping** with validation-set monitoring. Models with validation ROC-AUC below 0.6 are automatically excluded from the stacking ensemble.

### 5. Overfitting Detection & Mitigation

The pipeline includes an **explicit before/after comparison**:

- **Baseline**: XGBoost with no regularisation — deep trees (depth=12), no L1/L2, no subsampling
- **Regularised**: Production XGBoost with `reg_alpha=0.1`, `reg_lambda=1.0`, `subsample=0.8`, `colsample_bytree=0.8`, early stopping (patience=50), and moderate tree depth

The comparison is plotted as a side-by-side bar chart showing train vs. validation gaps. Additional overfitting diagnostics:
- Train/val metric comparison for every model
- Learning curves showing score convergence vs. training set size

### 6. Recall Optimisation (30%-60%)

For each target recall level (30%, 35%, 40%, 45%, 50%, 55%, 60%), the pipeline:

1. Sweeps the decision threshold across the validation PR curve
2. Finds the threshold achieving that recall with maximal precision
3. Reports actual recall, precision, and F1 at each operating point
4. Plots the operating points on the PR curve

This allows stakeholders to choose the precision-recall trade-off that matches their cost model.

### 7. Explainability

SHAP (SHapley Additive exPlanations) values are computed for the best model, producing:
- **Beeswarm plot**: Each feature's impact direction and magnitude on individual predictions
- **Bar chart**: Features ranked by mean absolute SHAP value (global importance)

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py

# Optional: run with hyperparameter tuning (~30 min)
# Linux/macOS:
TUNE_HYPERPARAMS=1 python main.py
# Windows (PowerShell):
$env:TUNE_HYPERPARAMS="1"; python main.py

# Optional: run with CV-based model selection (~10 min extra)
# Linux/macOS:
RUN_CV_SELECTION=1 python main.py
# Windows (PowerShell):
$env:RUN_CV_SELECTION="1"; python main.py


## Outputs

All results are saved in `output/`:

| File | Description |
|------|-------------|
| `best_model.joblib` | Serialised best classifier (deployment-ready) |
| `lookup_tables.joblib` | Feature engineering mappings (needed at inference) |
| `test_predictions.csv` | Fraud probabilities and binary predictions for test set |
| `recall_summary.csv` | Precision/recall/F1 at each target recall level |
| `shap_summary_*.png` | SHAP feature importance (beeswarm) |
| `shap_importance_*.png` | SHAP feature importance (bar chart) |
| `overfit_comparison_*.png` | Baseline vs. regularised overfitting comparison |
| `comparison_*.png` | Model comparison plots (ROC, PR, metrics) |
| `<Model>/` | Per-model plots (confusion matrix, PR curve, ROC, feature importance) |

## Project Structure

```
├── main.py                     Pipeline orchestrator
├── src/
│   ├── config.py               Paths, feature lists, constants
│   ├── data_quality.py         Data cleaning pipeline (6 steps)
│   ├── features.py             Feature engineering + lookup tables
│   ├── model.py                Model builders, training, SMOTE, tuning
│   └── evaluation.py           Metrics, plots, SHAP, recall optimisation
├── requirements.txt            Python dependencies
├── train_data.csv              Training data
├── test_data.csv               Test data
└── output/                     All results
```

## Dependencies

pandas, numpy, scikit-learn, imbalanced-learn, xgboost, lightgbm, matplotlib, seaborn, optuna, joblib, shap

See [requirements.txt](requirements.txt) for pinned versions.
