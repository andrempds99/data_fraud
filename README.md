# Fraud Detection ML Pipeline

End-to-end machine learning pipeline for **credit card fraud detection**. Trains multiple classifiers on historical transaction data, engineers domain-specific features, and selects the best model based on PR-AUC performance.

## Models

- **Logistic Regression** — baseline with class balancing
- **Random Forest** — 300 trees, balanced class weights
- **XGBoost** — 500 estimators, scale_pos_weight for imbalance
- **LightGBM** — 500 estimators, is_unbalance=True

Optional Optuna hyperparameter tuning for the best-performing model.

## Pipeline

1. **Data loading & quality** — normalize nulls, deduplicate, fix types
2. **Temporal split** — 80/20 by timestamp (no data leakage)
3. **Feature engineering** — 27 features including 7-day card velocity, amount deviation, merchant stats, geo mismatches, temporal features, and email domain frequency
4. **Model training** — fit all classifiers with preprocessing (imputation + scaling/encoding)
5. **Evaluation** — PR-AUC, ROC-AUC, F1, confusion matrices, PR/ROC curves, feature importance
6. **Best model selection** — serialized as `best_model.joblib`

## Project Structure

```
main.py                    # Main training pipeline
generate_report.py         # Full PDF report
generate_summary_report.py # 3-page summary PDF
src/
  config.py                # Paths, feature lists, constants
  data_quality.py          # Loading, cleaning, deduplication
  features.py              # Feature engineering & lookup tables
  model.py                 # Preprocessing, training, cross-validation
  evaluation.py            # Metrics, plots, threshold optimization
output/                    # Models, predictions, per-model results
```

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Train models and select best
python main.py

# With cross-validation ranking
RUN_CV_SELECTION=1 python main.py

# With Optuna hyperparameter tuning
TUNE_HYPERPARAMS=1 python main.py

# Generate reports
python generate_report.py
python generate_summary_report.py
```

## Output

- `best_model.joblib` — serialized best classifier
- `lookup_tables.joblib` — feature aggregation mappings for inference
- `test_predictions.csv` — fraud probabilities and binary predictions
- `recall_summary.csv` — precision/recall/F1 at target recall levels (30–60%)
- Per-model folders with confusion matrices, ROC/PR curves, and feature importance plots

## Dependencies

pandas, numpy, scikit-learn, imbalanced-learn, xgboost, lightgbm, matplotlib, seaborn, optuna, joblib

See [requirements.txt](requirements.txt) for pinned versions.
