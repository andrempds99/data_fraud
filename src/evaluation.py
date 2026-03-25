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
from sklearn.model_selection import learning_curve, TimeSeriesSplit

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


# ── SHAP explainability ───────────────────────────────────────────────────────

def plot_shap_summary(model, X_val, name="Model", max_display=20):
    """Compute SHAP values and save beeswarm + bar summary plots."""
    _ensure_output_dir()
    try:
        import shap
    except ImportError:
        logger.warning("  shap not installed (pip install shap), skipping.")
        return

    if not hasattr(model, "named_steps"):
        logger.info("  SHAP: model is not a Pipeline, skipping.")
        return

    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]
    X_t = preprocessor.transform(X_val)

    try:
        feat_names = list(preprocessor.get_feature_names_out())
    except AttributeError:
        feat_names = [f"f{i}" for i in range(X_t.shape[1])]

    logger.info("  Computing SHAP values for %s …", name)
    clf_type = type(classifier).__name__
    if clf_type in ("XGBClassifier", "LGBMClassifier",
                     "RandomForestClassifier", "GradientBoostingClassifier"):
        explainer = shap.TreeExplainer(classifier)
    else:
        logger.info("  SHAP: unsupported classifier %s, skipping.", clf_type)
        return

    shap_values = explainer.shap_values(X_t)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive (fraud) class

    # Beeswarm plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_t, feature_names=feat_names,
                      max_display=max_display, show=False)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"shap_summary_{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved %s", path)

    # Bar plot (mean |SHAP|)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_t, feature_names=feat_names,
                      plot_type="bar", max_display=max_display, show=False)
    plt.tight_layout()
    path_bar = os.path.join(OUTPUT_DIR, f"shap_importance_{name}.png")
    plt.savefig(path_bar, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved %s", path_bar)


# ── Overfitting comparison plot ────────────────────────────────────────────────

def plot_overfit_comparison(baseline_metrics, reg_metrics, name="XGBoost"):
    """Bar chart showing train/val gap for baseline vs regularised models."""
    _ensure_output_dir()
    labels = ["Train ROC-AUC", "Val ROC-AUC", "Train PR-AUC", "Val PR-AUC"]
    baseline = [baseline_metrics["train_roc"], baseline_metrics["val_roc"],
                baseline_metrics["train_pr"], baseline_metrics["val_pr"]]
    regularised = [reg_metrics["train_roc"], reg_metrics["val_roc"],
                   reg_metrics["train_pr"], reg_metrics["val_pr"]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, baseline, width,
                   label="Baseline (no regularisation)", color="#ff6b6b")
    bars2 = ax.bar(x + width / 2, regularised, width,
                   label="Regularised + early stopping", color="#4ecdc4")

    ax.set_ylabel("Score")
    ax.set_title(f"Overfitting Mitigation — {name}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.1)

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9)

    bl_gap = abs(baseline[0] - baseline[1])
    rg_gap = abs(regularised[0] - regularised[1])
    ax.text(0.02, 0.98,
            f"ROC-AUC gap:  baseline={bl_gap:.3f}   regularised={rg_gap:.3f}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"overfit_comparison_{name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("  Saved %s", path)


# ── SHAP dependence plots ─────────────────────────────────────────────────────

def plot_shap_dependence(model, X_val, name="Model", top_n=6):
    """SHAP dependence plots for the top-N most predictive features.

    For each top feature the x-axis shows the *original* (unscaled) feature
    value and the y-axis shows the SHAP contribution to the fraud score.
    Natural breakpoints in these scatter plots indicate where rule thresholds
    should be placed — regions where the SHAP value crosses zero correspond
    to the point at which a feature starts actively increasing fraud risk.

    Parameters
    ----------
    model : fitted sklearn Pipeline with 'preprocessor' and 'classifier' steps.
    X_val : pd.DataFrame — raw (unscaled) validation features.
    name  : str — used for output file naming and titles.
    top_n : int — number of top features to plot (arranged in a grid).
    """
    _ensure_output_dir()
    try:
        import shap
    except ImportError:
        logger.warning("  shap not installed (pip install shap), skipping dependence plots.")
        return

    if not hasattr(model, "named_steps"):
        logger.info("  SHAP dependence: model is not a Pipeline, skipping.")
        return

    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]
    clf_type = type(classifier).__name__

    if clf_type not in (
        "XGBClassifier", "LGBMClassifier",
        "RandomForestClassifier", "GradientBoostingClassifier",
    ):
        logger.info("  SHAP dependence: unsupported classifier %s, skipping.", clf_type)
        return

    X_t = preprocessor.transform(X_val)
    try:
        feat_names = list(preprocessor.get_feature_names_out())
    except AttributeError:
        feat_names = [f"f{i}" for i in range(X_t.shape[1])]

    logger.info("  Computing SHAP values for dependence plots (%s) …", name)
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_t)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # fraud class

    # Rank features by mean |SHAP|
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:top_n]

    ncols = 3
    nrows = (top_n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten()

    rng = np.random.default_rng(42)

    for plot_pos, feat_idx in enumerate(top_idx):
        ax = axes_flat[plot_pos]
        transformed_name = feat_names[feat_idx]
        shap_col = shap_values[:, feat_idx]

        # Use original (unscaled) values for numeric features so the x-axis
        # is immediately readable in business units.
        if transformed_name.startswith("num__"):
            orig_col = transformed_name[len("num__"):]
            if orig_col in X_val.columns:
                feat_vals = X_val[orig_col].values
                x_label = orig_col.replace("_", " ").title()
            else:
                feat_vals = X_t[:, feat_idx]
                x_label = transformed_name
        else:
            feat_vals = X_t[:, feat_idx]
            x_label = transformed_name.replace("num__", "").replace("cat__", "")

        # Subsample large arrays for plotting speed
        n = len(feat_vals)
        if n > 5000:
            sample_idx = rng.choice(n, size=5000, replace=False)
            fv_plot = feat_vals[sample_idx]
            sv_plot = shap_col[sample_idx]
        else:
            fv_plot = feat_vals
            sv_plot = shap_col

        ax.scatter(fv_plot, sv_plot, alpha=0.3, s=6, color="steelblue")
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel("SHAP value", fontsize=9)
        ax.set_title(x_label, fontsize=10, fontweight="bold")

    # Hide unused panels
    for ax in axes_flat[top_n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"SHAP Dependence Plots — {name}\n"
        "(y > 0 increases fraud score; zero-crossings indicate rule thresholds)",
        fontsize=11,
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"shap_dependence_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", path)


# ── F-beta threshold optimisation ─────────────────────────────────────────────

def optimize_threshold_fbeta(y_true, y_prob, betas=None):
    """Find the threshold maximising F_beta for multiple beta values.

    Beta > 1 weights recall more (e.g. beta=2: recall twice as important).
    Beta < 1 weights precision more (e.g. beta=0.5).

    Returns a DataFrame with one row per beta: optimal threshold, precision,
    recall, and F_beta.
    """
    if betas is None:
        betas = [0.5, 1.0, 2.0]

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns one extra element in precision/recall
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    rows = []
    for beta in betas:
        beta_sq = beta ** 2
        denom = beta_sq * precisions + recalls
        fbeta = np.where(denom > 0, (1 + beta_sq) * precisions * recalls / denom, 0.0)
        best_idx = np.argmax(fbeta)
        thr = thresholds[best_idx]
        y_pred = (y_prob >= thr).astype(int)
        rows.append({
            "beta":      beta,
            "threshold": round(float(thr), 6),
            "precision": round(precision_score(y_true, y_pred, zero_division=0.0), 4),
            "recall":    round(recall_score(y_true, y_pred, zero_division=0.0), 4),
            "f_beta":    round(float(fbeta[best_idx]), 4),
        })

    df = pd.DataFrame(rows)
    logger.info("  F-beta optimisation:\n%s", df.to_string(index=False))
    return df


# ── Calibration diagnostics ──────────────────────────────────────────────────

def plot_calibration_curve(y_true, y_prob_before, y_prob_after, name="Model"):
    """Reliability diagram + Brier score before/after probability calibration."""
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    _ensure_output_dir()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for label, probs, color in [
        ("Before calibration", y_prob_before, "tab:red"),
        ("After calibration",  y_prob_after,  "tab:blue"),
    ]:
        brier = brier_score_loss(y_true, probs)
        try:
            prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10, strategy="uniform")
        except ValueError:
            continue
        ax1.plot(prob_pred, prob_true, "s-", color=color,
                 label=f"{label}  (Brier={brier:.4f})")

    ax1.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_title(f"Reliability Diagram — {name}")
    ax1.legend(fontsize=8)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Histogram of predicted probabilities
    ax2.hist(y_prob_before, bins=50, alpha=0.5, label="Before", color="tab:red")
    ax2.hist(y_prob_after,  bins=50, alpha=0.5, label="After",  color="tab:blue")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Prediction Distribution — {name}")
    ax2.legend()
    ax2.set_yscale("log")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"calibration_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", path)


# ── Cost-sensitive evaluation ─────────────────────────────────────────────────

def compute_expected_cost(
    y_true,
    y_prob,
    cost_fn: float = 100.0,
    cost_fp: float = 1.0,
    thresholds=None,
) -> pd.DataFrame:
    """Compute expected cost at various thresholds using a cost matrix.

    Parameters
    ----------
    cost_fn : float — cost of a false negative (missed fraud). Default 100.
    cost_fp : float — cost of a false positive (false alert). Default 1.
    thresholds : optional array of thresholds; if None, uses linspace(0.01,0.99,50).

    Returns a DataFrame with threshold, expected cost, FP count, FN count, and
    the optimal threshold minimising total cost.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 50)

    rows = []
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        total_cost = fn * cost_fn + fp * cost_fp
        rows.append({
            "threshold":  round(float(thr), 4),
            "total_cost": total_cost,
            "fp":         int(fp),
            "fn":         int(fn),
            "tp":         int(tp),
            "tn":         int(tn),
        })

    df = pd.DataFrame(rows)
    best_row = df.loc[df["total_cost"].idxmin()]
    logger.info(
        "  Cost analysis (FN=$%.0f, FP=$%.0f): optimal threshold=%.4f  "
        "total cost=$%.0f  (FN=%d, FP=%d)",
        cost_fn, cost_fp,
        best_row["threshold"], best_row["total_cost"],
        best_row["fn"], best_row["fp"],
    )
    return df
