import logging
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import learning_curve, StratifiedKFold, TimeSeriesSplit

from src.config import OUTPUT_DIR, TARGET_RECALLS, SEED

logger = logging.getLogger(__name__)


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _model_dir(name):
    """Return and create a per-model subdirectory under OUTPUT_DIR."""
    d = os.path.join(OUTPUT_DIR, name)
    os.makedirs(d, exist_ok=True)
    return d


# ── Core metrics ───────────────────────────────────────────────────────────────

def print_metrics(model, X, y, label=""):
    """Print classification report + AUC scores."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    logger.info("--- %s ---", label)
    logger.info("\n%s", classification_report(y, y_pred, target_names=["Legit", "Fraud"]))
    logger.info("  ROC-AUC:  %.4f", roc_auc_score(y, y_prob))
    logger.info("  PR-AUC:   %.4f", average_precision_score(y, y_prob))
    return y_pred, y_prob


# ── Overfitting check ─────────────────────────────────────────────────────────

def check_overfit(model, X_train, y_train, X_val, y_val, name="Model"):
    """Compare train vs validation metrics to detect overfitting."""
    logger.info("=" * 60)
    logger.info("  Overfit Check: %s", name)
    logger.info("=" * 60)
    for split_name, X, y in [("Train", X_train, y_train), ("Val", X_val, y_val)]:
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)
        roc = roc_auc_score(y, y_prob)
        pr = average_precision_score(y, y_prob)
        f1 = f1_score(y, y_pred)
        logger.info("  %-6s  ROC-AUC=%.4f  PR-AUC=%.4f  F1=%.4f", split_name, roc, pr, f1)


# ── Confusion Matrix plot ─────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, name="Model"):
    _ensure_output_dir()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues",
        xticklabels=["Legit", "Fraud"],
        yticklabels=["Legit", "Fraud"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    path = os.path.join(_model_dir(name), f"confusion_matrix_{name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("  Saved %s", path)


# ── Precision-Recall curve ─────────────────────────────────────────────────────

def plot_precision_recall(y_true, y_prob, name="Model"):
    _ensure_output_dir()
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(recall, precision, lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {name}  (AP={ap:.3f})")
    ax.axhline(y=y_true.mean(), color="grey", linestyle="--", label="Baseline (fraud rate)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(_model_dir(name), f"pr_curve_{name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("  Saved %s", path)


# ── ROC curve ──────────────────────────────────────────────────────────────────

def plot_roc(y_true, y_prob, name="Model"):
    _ensure_output_dir()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(f"ROC Curve — {name}")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(_model_dir(name), f"roc_{name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("  Saved %s", path)


# ── Feature importance ─────────────────────────────────────────────────────────

def plot_feature_importance(model, name="Model", top_n=20):
    _ensure_output_dir()
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        logger.info("  %s: no feature_importances_ attribute, skipping.", name)
        return

    importances = classifier.feature_importances_
    try:
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    except AttributeError:
        feature_names = [f"f{i}" for i in range(len(importances))]

    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    feat_imp = feat_imp.tail(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    feat_imp.plot(kind="barh", ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances — {name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = os.path.join(_model_dir(name), f"feature_importance_{name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("  Saved %s", path)


# ── Learning curves ────────────────────────────────────────────────────────────

def plot_learning_curves(model, X, y, name="Model", cv_folds=5):
    _ensure_output_dir()
    logger.info("  Computing learning curves for %s (this may take a while) …", name)
    # TimeSeriesSplit respects the temporal ordering of the data, consistent
    # with the temporal train/val split used throughout the pipeline.
    cv = TimeSeriesSplit(n_splits=cv_folds)
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=cv,
        scoring="average_precision",
        n_jobs=-1,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_scores.mean(axis=1), label="Train")
    ax.plot(train_sizes, val_scores.mean(axis=1), label="Validation")
    ax.fill_between(
        train_sizes,
        train_scores.mean(axis=1) - train_scores.std(axis=1),
        train_scores.mean(axis=1) + train_scores.std(axis=1),
        alpha=0.1,
    )
    ax.fill_between(
        train_sizes,
        val_scores.mean(axis=1) - val_scores.std(axis=1),
        val_scores.mean(axis=1) + val_scores.std(axis=1),
        alpha=0.1,
    )
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Average Precision")
    ax.set_title(f"Learning Curves — {name}")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(_model_dir(name), f"learning_curves_{name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("  Saved %s", path)


# ── Recall optimisation ───────────────────────────────────────────────────────

def find_recall_thresholds(y_true, y_prob, target_recalls=None):
    """Find the threshold that achieves each target recall, returning a DataFrame."""
    if target_recalls is None:
        target_recalls = TARGET_RECALLS

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    rows = []

    for target in target_recalls:
        valid_idx = np.where(recalls >= target)[0]
        if len(valid_idx) == 0:
            continue
        best_idx = valid_idx[np.argmax(precisions[valid_idx])]
        thr = thresholds[min(best_idx, len(thresholds) - 1)]

        y_pred = (y_prob >= thr).astype(int)
        rows.append({
            "target_recall": target,
            "threshold": thr,
            "actual_recall": recall_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
        })

    df = pd.DataFrame(rows)
    return df


def plot_recall_operating_points(y_true, y_prob, name="Model", target_recalls=None):
    """Plot the PR curve with target recall operating points marked."""
    _ensure_output_dir()
    if target_recalls is None:
        target_recalls = TARGET_RECALLS

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(recalls, precisions, lw=2)

    for target in target_recalls:
        valid_idx = np.where(recalls >= target)[0]
        if len(valid_idx) == 0:
            continue
        best_idx = valid_idx[np.argmax(precisions[valid_idx])]
        ax.plot(recalls[best_idx], precisions[best_idx], "ro", markersize=8)
        ax.annotate(
            f"R={target:.0%}",
            (recalls[best_idx], precisions[best_idx]),
            textcoords="offset points", xytext=(10, 5), fontsize=9,
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Operating Points at Target Recalls — {name}")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    path = os.path.join(_model_dir(name), f"recall_operating_points_{name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("  Saved %s", path)


# ── Full evaluation for one model ─────────────────────────────────────────────

def evaluate_model(model, X_train, y_train, X_val, y_val, name="Model"):
    """Run all evaluation steps for a single model."""
    logger.info("#" * 60)
    logger.info("  EVALUATION: %s", name)
    logger.info("#" * 60)

    # Classification report + AUC
    y_pred, y_prob = print_metrics(model, X_val, y_val, label=f"{name} – Validation")

    # Overfit check
    check_overfit(model, X_train, y_train, X_val, y_val, name)

    # Plots
    plot_confusion_matrix(y_val, y_pred, name)
    plot_precision_recall(y_val, y_prob, name)
    plot_roc(y_val, y_prob, name)
    plot_feature_importance(model, name)

    # Recall operating points
    recall_df = find_recall_thresholds(y_val, y_prob)
    logger.info("  Recall Operating Points (%s):\n%s", name, recall_df.to_string(index=False))
    plot_recall_operating_points(y_val, y_prob, name)

    return y_prob, recall_df


# ── Model comparison plots ────────────────────────────────────────────────────

def plot_roc_comparison(y_true, model_probs):
    """Overlay ROC curves for all models on a single figure."""
    _ensure_output_dir()
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, y_prob in model_probs.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve — Model Comparison")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "comparison_roc.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("  Saved %s", path)


def plot_pr_comparison(y_true, model_probs):
    """Overlay Precision-Recall curves for all models on a single figure."""
    _ensure_output_dir()
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, y_prob in model_probs.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, lw=2, label=f"{name} (AP={ap:.3f})")
    ax.axhline(y=y_true.mean(), color="grey", linestyle="--", label="Baseline (fraud rate)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Model Comparison")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "comparison_pr_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("  Saved %s", path)


def plot_metrics_comparison(y_true, model_probs):
    """Bar chart comparing ROC-AUC, PR-AUC, and F1 across models."""
    _ensure_output_dir()
    rows = []
    for name, y_prob in model_probs.items():
        y_pred = (y_prob >= 0.5).astype(int)
        rows.append({
            "Model": name,
            "ROC-AUC": roc_auc_score(y_true, y_prob),
            "PR-AUC": average_precision_score(y_true, y_prob),
            "F1": f1_score(y_true, y_pred),
        })
    df = pd.DataFrame(rows).set_index("Model")

    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(kind="bar", ax=ax, rot=0)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Key Metrics")
    ax.set_ylim([0, 1])
    ax.legend(loc="lower right")
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "comparison_metrics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("  Saved %s", path)
