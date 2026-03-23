"""
Fraud Detection Pipeline — End-to-End Runner
=============================================

Usage:
    python main.py

Optional environment flags:
    TUNE_HYPERPARAMS=1     Run Optuna LightGBM tuning after initial training.
    RUN_CV_SELECTION=1     Run 3-fold TimeSeriesSplit CV to rank models before
                           selecting the best (adds ~10 min but is more robust).

Outputs are saved in the ``output/`` directory:
  - Plots (confusion matrix, PR curve, ROC, feature importance, learning curves)
  - ``best_model.joblib``      fitted model (reload with joblib.load)
  - ``lookup_tables.joblib``   aggregation mappings used during feature engineering
  - ``test_predictions.csv``   fraud probabilities and binary predictions
  - ``recall_summary.csv``     precision/recall/F1 at each target recall level
"""

import logging
import os
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score

# Suppress only well-understood, benign warnings from third-party libraries.
# Do NOT use a blanket category-level suppression — it hides real issues.
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", message=".*No positive samples.*", category=UserWarning)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

from src.config import OUTPUT_DIR, SEED
from src.data_quality import load_data, run_quality_pipeline
from src.features import run_feature_pipeline
from src.model import (
    split_data_temporal,
    train_all_models,
    tune_lightgbm,
    cross_validate_temporal,
)
from src.evaluation import evaluate_model, find_recall_thresholds, plot_learning_curves, plot_roc_comparison, plot_pr_comparison, plot_metrics_comparison

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TUNE_HYPERPARAMS   = os.environ.get("TUNE_HYPERPARAMS",   "0") == "1"
RUN_CV_SELECTION   = os.environ.get("RUN_CV_SELECTION",   "0") == "1"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    run_start = datetime.now(tz=timezone.utc)
    logger.info("=" * 60)
    logger.info("  RUN STARTED: %s", run_start.isoformat())
    logger.info("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  STEP 1: LOADING DATA")
    logger.info("=" * 60)
    train, test, y_train = load_data()

    # ── 2. Data quality ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  STEP 2: DATA QUALITY")
    logger.info("=" * 60)
    train, test, y_train, test_ids = run_quality_pipeline(train, test, y_train)

    # ── 3. Train/val split — BEFORE feature engineering (prevents leakage) ─
    logger.info("=" * 60)
    logger.info("  STEP 3: TEMPORAL TRAIN / VALIDATION SPLIT")
    logger.info("=" * 60)
    X_train_raw, X_val_raw, y_tr, y_val = split_data_temporal(train, y_train)

    # ── 4. Feature engineering ────────────────────────────────────────────
    # Lookup tables built from X_train_raw only — val and test rows never
    # influence training statistics.
    logger.info("=" * 60)
    logger.info("  STEP 4: FEATURE ENGINEERING")
    logger.info("=" * 60)
    X_train_fe, extra_fes, lookup_tables = run_feature_pipeline(
        X_train_raw, extra_dfs=[X_val_raw, test]
    )
    X_val_fe, test_fe = extra_fes

    # ── Column alignment assertion ────────────────────────────────────────
    train_cols      = set(X_train_fe.columns)
    missing_in_val  = train_cols - set(X_val_fe.columns)
    extra_in_val    = set(X_val_fe.columns) - train_cols
    missing_in_test = train_cols - set(test_fe.columns)
    extra_in_test   = set(test_fe.columns) - train_cols
    assert not missing_in_val and not extra_in_val, (
        f"Train/val column mismatch — train only: {missing_in_val}; "
        f"val only: {extra_in_val}"
    )
    assert not missing_in_test and not extra_in_test, (
        f"Train/test column mismatch — train only: {missing_in_test}; "
        f"test only: {extra_in_test}"
    )
    logger.info("Column alignment OK: %d features", len(train_cols))

    # ── 5. Train models ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  STEP 5: TRAINING CLASSIFIERS")
    logger.info("=" * 60)
    models = train_all_models(X_train_fe, y_tr)

    # ── 6. Evaluate all models + optional CV ranking ───────────────────
    logger.info("=" * 60)
    logger.info("  STEP 6: EVALUATION & METRICS")
    logger.info("=" * 60)
    best_model_name = None
    best_pr_auc = -1.0
    all_recall_dfs = []
    all_model_probs = {}

    for name, model in models.items():
        y_prob, recall_df = evaluate_model(model, X_train_fe, y_tr, X_val_fe, y_val, name)
        recall_df["model"] = name
        pr_auc = average_precision_score(y_val, y_prob)
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_model_name = name
        all_recall_dfs.append(recall_df)
        all_model_probs[name] = y_prob

    # ── Comparison plots across all models ─────────────────────────────
    plot_roc_comparison(y_val, all_model_probs)
    plot_pr_comparison(y_val, all_model_probs)
    plot_metrics_comparison(y_val, all_model_probs)

    # ── 6b. CV-based model ranking (gated — set RUN_CV_SELECTION=1 to enable) ─
    # Using TimeSeriesSplit is more robust than single-split selection but adds
    # significant runtime (≈10 min for all 4 models × 3 folds).
    if RUN_CV_SELECTION:
        logger.info("=" * 60)
        logger.info("  STEP 6b: TimeSeriesSplit CV RANKING (3-fold)")
        logger.info("=" * 60)
        cv_scores = {}
        for name, model in models.items():
            scores = cross_validate_temporal(clone(model), X_train_fe, y_tr, cv_folds=3)
            cv_scores[name] = scores.mean()
            logger.info("  %-25s CV PR-AUC = %.4f ± %.4f", name, scores.mean(), scores.std())
        # Override best-model selection with CV-based ranking.
        best_model_name = max(cv_scores, key=cv_scores.get)
        best_pr_auc = cv_scores[best_model_name]
        logger.info("  BEST (CV-based): %s  (CV PR-AUC = %.4f)", best_model_name, best_pr_auc)
    else:
        logger.info(
            "  Tip: set RUN_CV_SELECTION=1 for TimeSeriesSplit CV-based model selection."
        )

    recall_summary = pd.concat(all_recall_dfs, ignore_index=True)
    recall_path = os.path.join(OUTPUT_DIR, "recall_summary.csv")
    recall_summary.to_csv(recall_path, index=False)
    logger.info("Recall summary saved to %s", recall_path)

    # ── 7. Best model selection ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  BEST MODEL: %s  (PR-AUC = %.4f)", best_model_name, best_pr_auc)
    logger.info("=" * 60)
    best_model = models[best_model_name]

    # ── 7b. Optional Optuna tuning (env: TUNE_HYPERPARAMS=1) ─────────────
    if TUNE_HYPERPARAMS:
        logger.info("=" * 60)
        logger.info("  STEP 7b: OPTUNA HYPERPARAMETER TUNING (n_trials=50)")
        logger.info("=" * 60)
        tuned_model = tune_lightgbm(X_train_fe, y_tr, n_trials=50)
        tuned_prob  = tuned_model.predict_proba(X_val_fe)[:, 1]
        tuned_pr_auc = average_precision_score(y_val, tuned_prob)
        logger.info(
            "Tuned PR-AUC: %.4f  vs  baseline best: %.4f",
            tuned_pr_auc, best_pr_auc,
        )
        if tuned_pr_auc > best_pr_auc:
            best_model = tuned_model
            best_pr_auc = tuned_pr_auc
            best_model_name = "LightGBM_Tuned"
            logger.info("Switching to Optuna-tuned model as best.")

    # ── 8. Serialise best model + lookup tables ───────────────────────────
    model_path  = os.path.join(OUTPUT_DIR, "best_model.joblib")
    lookup_path = os.path.join(OUTPUT_DIR, "lookup_tables.joblib")
    joblib.dump(best_model, model_path)
    joblib.dump(lookup_tables, lookup_path)
    logger.info("Best model saved to %s", model_path)
    logger.info("Lookup tables saved to %s", lookup_path)

    # ── 9. Learning curves for best model ────────────────────────────────
    logger.info("=" * 60)
    logger.info("  STEP 9: LEARNING CURVES (best model)")
    logger.info("=" * 60)
    plot_learning_curves(best_model, X_train_fe, y_tr, name=best_model_name, cv_folds=3)

    # ── 10. Generate test predictions ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  STEP 10: TEST PREDICTIONS")
    logger.info("=" * 60)
    y_val_prob = best_model.predict_proba(X_val_fe)[:, 1]
    recall_df  = find_recall_thresholds(y_val, y_val_prob)
    target_row = recall_df.loc[(recall_df["target_recall"] - 0.50).abs().idxmin()]
    chosen_threshold = target_row["threshold"]
    logger.info(
        "Chosen threshold: %.4f  (target recall ~50%%, actual=%.2f%%, precision=%.2f%%)",
        chosen_threshold,
        target_row["actual_recall"] * 100,
        target_row["precision"] * 100,
    )

    test_probs = best_model.predict_proba(test_fe)[:, 1]
    test_preds = (test_probs >= chosen_threshold).astype(int)

    # test_ids were sanitized (quotes/commas removed) in run_quality_pipeline.
    safe_ids = test_ids.values if test_ids is not None else range(len(test_fe))

    output = pd.DataFrame({
        "transactionid":     safe_ids,
        "fraud_probability": test_probs,
        "fraud_prediction":  test_preds,
    })
    pred_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
    output.to_csv(pred_path, index=False)
    logger.info("Predictions saved to %s", pred_path)
    logger.info("Predicted fraud: %d / %d (%.2f%%)",
        test_preds.sum(), len(test_preds), test_preds.mean() * 100,
    )

    run_end = datetime.now(tz=timezone.utc)
    logger.info("=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info("  Run finished: %s", run_end.isoformat())
    logger.info("  Elapsed:      %.1f s", (run_end - run_start).total_seconds())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

