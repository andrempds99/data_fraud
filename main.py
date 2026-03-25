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
from sklearn.metrics import average_precision_score, roc_auc_score

# Suppress only well-understood, benign warnings from third-party libraries.
# Do NOT use a blanket category-level suppression — it hides real issues.
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", message=".*No positive samples.*", category=UserWarning)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)
# Learning curves on a calibrated model triggered when tiny training folds
# have fewer than 3 fraud examples — expected and non-fatal.
warnings.filterwarnings(
    "ignore",
    message=".*fits failed.*",
    category=UserWarning,
)

from src.config import OUTPUT_DIR, SEED, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.data_quality import load_data, run_quality_pipeline
from src.features import run_feature_pipeline
from src.model import (
    split_data_temporal,
    train_all_models,
    tune_lightgbm,
    tune_xgboost,
    cross_validate_temporal,
    build_stacking_ensemble,
    calibrate_model,
    train_with_smote,
    train_with_undersampling,
    build_baseline_xgboost,
    build_preprocessor,
)
from src.evaluation import (
    evaluate_model,
    find_recall_thresholds,
    optimize_threshold_fbeta,
    plot_learning_curves,
    plot_roc_comparison,
    plot_pr_comparison,
    plot_metrics_comparison,
    plot_shap_summary,
    plot_shap_dependence,
    plot_overfit_comparison,
    plot_calibration_curve,
    compute_expected_cost,
)
from src.rules import run_rule_extraction
from src.output_manager import reset_output_dir, organize_output_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TUNE_HYPERPARAMS   = os.environ.get("TUNE_HYPERPARAMS",   "0") == "1"
RUN_CV_SELECTION   = os.environ.get("RUN_CV_SELECTION",   "0") == "1"


def main():
    # Start from a clean slate so each run only contains fresh artifacts.
    reset_output_dir(OUTPUT_DIR)

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
        X_train_raw, extra_dfs=[X_val_raw, test], y_train=y_tr,
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
    models = train_all_models(X_train_fe, y_tr, X_val_fe, y_val)

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

    # ── 6a. Stacking ensemble (filter underperforming models) ───────────
    logger.info("=" * 60)
    logger.info("  STEP 6a: STACKING ENSEMBLE")
    logger.info("=" * 60)
    MIN_ROC_AUC_STACK = 0.6
    stacking_models = {}
    for sname, smodel in models.items():
        val_prob_s = smodel.predict_proba(X_val_fe)[:, 1]
        roc_s = roc_auc_score(y_val, val_prob_s)
        if roc_s >= MIN_ROC_AUC_STACK:
            stacking_models[sname] = smodel
        else:
            logger.warning("  Excluding %s from stacking (ROC-AUC=%.4f < %.2f)",
                           sname, roc_s, MIN_ROC_AUC_STACK)
    if len(stacking_models) < 2:
        logger.warning("  Fewer than 2 models passed filter; using all models for stacking.")
        stacking_models = models
    stacking_model = build_stacking_ensemble(
        stacking_models, X_train_fe, y_tr, X_val_fe, y_val
    )
    stack_prob = stacking_model.predict_proba(X_val_fe)[:, 1]
    stack_pr_auc = average_precision_score(y_val, stack_prob)
    logger.info("  Stacking PR-AUC: %.4f", stack_pr_auc)
    all_model_probs["Stacking"] = stack_prob
    if stack_pr_auc > best_pr_auc:
        best_pr_auc = stack_pr_auc
        best_model_name = "Stacking"

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

    # ── 6c. Resampling comparison (SMOTE + Undersampling on all models) ────
    logger.info("=" * 60)
    logger.info("  STEP 6c: RESAMPLING COMPARISON (ALL MODELS)")
    logger.info("=" * 60)
    smote_wins = False
    for name, model in models.items():
        if not hasattr(model, "named_steps"):
            continue
        base_prob = model.predict_proba(X_val_fe)[:, 1]
        base_pr = average_precision_score(y_val, base_prob)

        smote_m = train_with_smote(model, X_train_fe, y_tr, name)
        smote_prob = smote_m.predict_proba(X_val_fe)[:, 1]
        smote_pr = average_precision_score(y_val, smote_prob)

        under_m = train_with_undersampling(model, X_train_fe, y_tr, name)
        under_prob = under_m.predict_proba(X_val_fe)[:, 1]
        under_pr = average_precision_score(y_val, under_prob)

        logger.info("  %-20s  base=%.4f  SMOTE=%.4f  undersample=%.4f",
                     name, base_pr, smote_pr, under_pr)

        best_variant_pr = max(base_pr, smote_pr, under_pr)
        if best_variant_pr > best_pr_auc:
            if smote_pr == best_variant_pr:
                best_pr_auc = smote_pr
                best_model_name = name
                models[name] = smote_m
                smote_wins = True
                logger.info("  → %s SMOTE improves PR-AUC to %.4f", name, smote_pr)
            elif under_pr == best_variant_pr:
                best_pr_auc = under_pr
                best_model_name = name
                models[name] = under_m
                smote_wins = True
                logger.info("  → %s Undersampling improves PR-AUC to %.4f", name, under_pr)

    if not smote_wins:
        logger.info("  → Class weights outperform resampling; keeping original models.")

    # ── 7. Best model selection ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  BEST MODEL: %s  (PR-AUC = %.4f)", best_model_name, best_pr_auc)
    logger.info("=" * 60)
    if best_model_name == "Stacking":
        best_model = stacking_model
    else:
        best_model = models[best_model_name]

    # ── 7a. Overfitting mitigation demonstration ───────────────────────
    logger.info("=" * 60)
    logger.info("  STEP 7a: OVERFITTING MITIGATION COMPARISON")
    logger.info("=" * 60)
    num_feats = [c for c in NUMERIC_FEATURES if c in X_train_fe.columns]
    cat_feats = [c for c in CATEGORICAL_FEATURES if c in X_train_fe.columns]
    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    baseline_pipe = build_baseline_xgboost(
        build_preprocessor(num_feats, cat_feats), pos_weight
    )
    baseline_pipe.fit(X_train_fe, y_tr)
    bl_train_prob = baseline_pipe.predict_proba(X_train_fe)[:, 1]
    bl_val_prob = baseline_pipe.predict_proba(X_val_fe)[:, 1]
    baseline_m = {
        "train_roc": roc_auc_score(y_tr, bl_train_prob),
        "val_roc":   roc_auc_score(y_val, bl_val_prob),
        "train_pr":  average_precision_score(y_tr, bl_train_prob),
        "val_pr":    average_precision_score(y_val, bl_val_prob),
    }
    reg_model_for_cmp = models.get("XGBoost", models[best_model_name])
    rg_train_prob = reg_model_for_cmp.predict_proba(X_train_fe)[:, 1]
    rg_val_prob = reg_model_for_cmp.predict_proba(X_val_fe)[:, 1]
    reg_m = {
        "train_roc": roc_auc_score(y_tr, rg_train_prob),
        "val_roc":   roc_auc_score(y_val, rg_val_prob),
        "train_pr":  average_precision_score(y_tr, rg_train_prob),
        "val_pr":    average_precision_score(y_val, rg_val_prob),
    }
    logger.info("  Baseline  (no reg)   — Train ROC=%.4f  Val ROC=%.4f  gap=%.4f",
                baseline_m["train_roc"], baseline_m["val_roc"],
                baseline_m["train_roc"] - baseline_m["val_roc"])
    logger.info("  Regularised          — Train ROC=%.4f  Val ROC=%.4f  gap=%.4f",
                reg_m["train_roc"], reg_m["val_roc"],
                reg_m["train_roc"] - reg_m["val_roc"])
    plot_overfit_comparison(baseline_m, reg_m, name="XGBoost")

    # ── 7b. Optional Optuna tuning (env: TUNE_HYPERPARAMS=1) ─────────────
    if TUNE_HYPERPARAMS:
        logger.info("=" * 60)
        logger.info("  STEP 7b: OPTUNA HYPERPARAMETER TUNING (n_trials=50)")
        logger.info("=" * 60)

        # Tune LightGBM
        tuned_lgb = tune_lightgbm(X_train_fe, y_tr, n_trials=50)
        tuned_lgb_prob  = tuned_lgb.predict_proba(X_val_fe)[:, 1]
        tuned_lgb_pr_auc = average_precision_score(y_val, tuned_lgb_prob)
        logger.info("Tuned LightGBM PR-AUC: %.4f", tuned_lgb_pr_auc)

        # Tune XGBoost
        tuned_xgb = tune_xgboost(X_train_fe, y_tr, n_trials=50)
        tuned_xgb_prob  = tuned_xgb.predict_proba(X_val_fe)[:, 1]
        tuned_xgb_pr_auc = average_precision_score(y_val, tuned_xgb_prob)
        logger.info("Tuned XGBoost PR-AUC: %.4f", tuned_xgb_pr_auc)

        # Pick the best tuned model
        if tuned_lgb_pr_auc >= tuned_xgb_pr_auc and tuned_lgb_pr_auc > best_pr_auc:
            best_model = tuned_lgb
            best_pr_auc = tuned_lgb_pr_auc
            best_model_name = "LightGBM_Tuned"
            logger.info("Switching to Optuna-tuned LightGBM as best.")
        elif tuned_xgb_pr_auc > best_pr_auc:
            best_model = tuned_xgb
            best_pr_auc = tuned_xgb_pr_auc
            best_model_name = "XGBoost_Tuned"
            logger.info("Switching to Optuna-tuned XGBoost as best.")

    # ── 7c. SHAP explainability ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  STEP 7c: SHAP EXPLAINABILITY")
    logger.info("=" * 60)
    plot_shap_summary(best_model, X_val_fe, name=best_model_name)
    plot_shap_dependence(best_model, X_val_fe, name=best_model_name)

    # ── 7d. Probability calibration ──────────────────────────────────────
    # Split validation into calibration (first 50%) and evaluation (last 50%)
    # so the calibration model never sees the data it's evaluated on.
    logger.info("=" * 60)
    logger.info("  STEP 7d: PROBABILITY CALIBRATION")
    logger.info("=" * 60)
    cal_split = len(X_val_fe) // 2
    X_cal, X_eval = X_val_fe.iloc[:cal_split], X_val_fe.iloc[cal_split:]
    y_cal, y_eval = y_val.iloc[:cal_split], y_val.iloc[cal_split:]
    logger.info("  Calibration split: Cal=%d  Eval=%d", len(X_cal), len(X_eval))

    y_prob_before_cal = best_model.predict_proba(X_eval)[:, 1]
    best_model = calibrate_model(best_model, X_cal, y_cal, method="isotonic")
    y_prob_after_cal = best_model.predict_proba(X_eval)[:, 1]
    plot_calibration_curve(y_eval, y_prob_before_cal, y_prob_after_cal, name=best_model_name)

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

    # ── 10a. F-beta threshold optimisation ───────────────────────────────
    logger.info("=" * 60)
    logger.info("  STEP 10a: F-BETA THRESHOLD OPTIMISATION")
    logger.info("=" * 60)
    fbeta_df = optimize_threshold_fbeta(y_val, y_val_prob, betas=[0.5, 1.0, 2.0])
    fbeta_path = os.path.join(OUTPUT_DIR, "fbeta_thresholds.csv")
    fbeta_df.to_csv(fbeta_path, index=False)
    logger.info("F-beta thresholds saved to %s", fbeta_path)

    # ── 10b. Cost-sensitive evaluation ───────────────────────────────────
    logger.info("=" * 60)
    logger.info("  STEP 10b: COST-SENSITIVE EVALUATION")
    logger.info("=" * 60)
    cost_df = compute_expected_cost(y_val, y_val_prob, cost_fn=100.0, cost_fp=1.0)
    cost_path = os.path.join(OUTPUT_DIR, "cost_analysis.csv")
    cost_df.to_csv(cost_path, index=False)
    logger.info("Cost analysis saved to %s", cost_path)

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

    # ── 11. Rule extraction ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  STEP 11: RULE EXTRACTION")
    logger.info("=" * 60)
    run_rule_extraction(
        best_model=best_model,
        best_model_name=best_model_name,
        X_train=X_train_fe,
        y_train=y_tr,
        X_val=X_val_fe,
        y_val=y_val,
        recall_threshold=chosen_threshold,
        all_models=models,
    )

    # Final pass: relocate files/folders into a consistent output layout.
    organize_output_dir(OUTPUT_DIR)

    run_end = datetime.now(tz=timezone.utc)
    logger.info("=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info("  Run finished: %s", run_end.isoformat())
    logger.info("  Elapsed:      %.1f s", (run_end - run_start).total_seconds())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

