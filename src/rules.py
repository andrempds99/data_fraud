"""Fraud rule extraction: surrogate trees, multi-depth sweep, combined coverage.

Workflow
--------
1. For each model, build surrogate decision trees at multiple depths (3-6)
   using BOTH model-predicted pseudo-labels AND true fraud labels, producing
   diverse candidate rules with thresholds in original business units
   (e.g. "Card Txn Count (last 7 days) > 15 AND Transaction Amount (EUR) > 500").

2. Simplify conditions (merge redundant bounds on the same feature) and format
   binary features as "Yes / No" instead of numeric thresholds.

3. Score every candidate rule on held-out validation data: precision, recall,
   F1, lift, coverage, and absolute fraud-hit count.

4. Compute combined OR-coverage: how much fraud the top-N rules catch together.

5. Persist:
   - ``output/surrogate_tree.txt``            — best surrogate tree dump
   - ``output/rules_summary.csv``             — scored rules ranked by lift
   - ``output/rules_combined_coverage.csv``   — cumulative OR-coverage table
   - ``output/rules_performance_*.png``       — precision-recall bubble chart
   - ``output/rules_leaderboard_*.png``       — top-N lift bar chart
   - ``output/rules_cumulative_recall_*.png`` — recall growth as rules added
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier, _tree, export_text

from src.config import OUTPUT_DIR, SEED

logger = logging.getLogger(__name__)


# ── Human-readable feature name mapping ───────────────────────────────────────

FEATURE_DISPLAY_NAMES: dict = {
    "euramount":                           "Transaction Amount (EUR)",
    "euramount_capped":                    "Transaction Amount Capped (EUR)",
    "log_euramount":                       "Log Transaction Amount",
    "card_txn_1d":                         "Card Txn Count (last 1 day)",
    "card_txn_3d":                         "Card Txn Count (last 3 days)",
    "card_txn_7d":                         "Card Txn Count (last 7 days)",
    "card_txn_14d":                        "Card Txn Count (last 14 days)",
    "card_txn_30d":                        "Card Txn Count (last 30 days)",
    "hours_since_last_txn":                "Hours Since Last Card Transaction",
    "card_avg_amount":                     "Card Average Amount (EUR)",
    "amount_deviation":                    "Amount Deviation from Card Avg (EUR)",
    "merchant_txn_count":                  "Merchant Transaction Count",
    "hour":                                "Transaction Hour",
    "dayofweek":                           "Day of Week (0=Mon)",
    "day":                                 "Day of Month",
    "is_weekend":                          "Is Weekend",
    "card_billing_mismatch":               "Card/Billing Country Mismatch",
    "geoip_billing_mismatch":              "GeoIP/Billing Country Mismatch",
    "domain_freq":                         "Email Domain Frequency",
    "cvvused":                             "CVV Used",
    "recurring":                           "Recurring Transaction",
    "initialrecurring":                    "Initial Recurring",
    "threedsused":                         "3DS Used",
    "success":                             "Transaction Success",
    "cardholder_disposabledomain_boolean": "Disposable Email Domain",
    "card_merchant_first":                 "First Time at This Merchant",
    "card_unique_merchants":               "Unique Merchants (card lifetime)",
}

# Features whose values are 0/1 flags — thresholds shown as integers
_BINARY_FEATURES = frozenset({
    "is_weekend",
    "card_billing_mismatch",
    "geoip_billing_mismatch",
    "cvvused",
    "recurring",
    "initialrecurring",
    "threedsused",
    "success",
    "cardholder_disposabledomain_boolean",
    "card_merchant_first",
})


# ── Text helpers ───────────────────────────────────────────────────────────────

def _display_name(feat: str) -> str:
    """Return a business-readable label for a feature column name."""
    return FEATURE_DISPLAY_NAMES.get(feat, feat.replace("_", " ").title())


def _format_condition(feat: str, op: str, val: float) -> str:
    """Format a single condition as a human-readable string.

    Binary features are displayed as '= Yes' / '= No' instead of numeric
    thresholds like '<= 0.5'.  Returns ``None`` for trivially-true conditions
    on binary features (e.g. '<= 0.5' on a 0/1 column always holds).
    """
    label = _display_name(feat)
    if feat in _BINARY_FEATURES:
        if op == "<=" and val >= 0.5:
            return None  # trivially true for 0/1 values → drop
        if op == ">" and val >= 0.5:
            return f"{label} = Yes"
        if op == "<=" and val < 0.5:
            return f"{label} = No"
        if op == ">" and val < 0.5:
            return f"{label} = Yes"
    if val == int(val) and abs(val) < 1e6:
        formatted = str(int(val))
    else:
        formatted = f"{val:.2f}"
    return f"{label} {op} {formatted}"


def _simplify_conditions(conditions: list) -> list:
    """Merge redundant conditions on the same feature.

    Multiple '>' splits on the same feature → keep the largest threshold.
    Multiple '<=' splits on the same feature → keep the smallest threshold.
    """
    lower: dict = {}  # feat → max of ">" thresholds
    upper: dict = {}  # feat → min of "<=" thresholds
    for feat, op, val in conditions:
        if op == ">":
            if feat not in lower or val > lower[feat]:
                lower[feat] = val
        else:
            if feat not in upper or val < upper[feat]:
                upper[feat] = val
    simplified = []
    seen = set()
    for feat, op, _val in conditions:
        key = (feat, op)
        if key in seen:
            continue
        seen.add(key)
        if op == ">" and feat in lower:
            simplified.append((feat, ">", lower[feat]))
        elif op == "<=" and feat in upper:
            simplified.append((feat, "<=", upper[feat]))
    return simplified


def _conditions_to_text(conditions: list) -> str:
    """Render simplified, human-readable conditions joined by AND."""
    simplified = _simplify_conditions(conditions)
    parts = []
    for feat, op, val in simplified:
        text = _format_condition(feat, op, val)
        if text is not None:
            parts.append(text)
    return " AND ".join(parts) if parts else "(no conditions)"


# ── Surrogate decision tree ────────────────────────────────────────────────────

def build_surrogate(
    X_train: pd.DataFrame,
    labels: np.ndarray,
    max_depth: int = 4,
    min_samples_leaf: int = 30,
) -> tuple:
    """Train a shallow DecisionTreeClassifier on raw (unscaled) numeric features.

    Only numeric columns are used so that every split threshold remains in its
    original business unit (e.g. "Card Txn Count (last 7 days) > 15").

    Parameters
    ----------
    X_train         : pd.DataFrame — raw feature matrix (pre-preprocessor).
    labels          : array-like — target labels (pseudo or true).
    max_depth       : int — depth cap (3–6 for readable rules).
    min_samples_leaf: int — minimum samples per leaf.

    Returns
    -------
    (surrogate, numeric_cols) — fitted DecisionTreeClassifier and column names.
    """
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X_train[numeric_cols]

    surrogate = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=SEED,
    )
    surrogate.fit(X_numeric, labels)
    return surrogate, numeric_cols


# ── Rule extraction from a fitted tree ────────────────────────────────────────

def _extract_leaf_rules(tree_clf: DecisionTreeClassifier, feature_names: list) -> list:
    """Walk every root-to-leaf path; return fraud-majority leaves as rules.

    Returns
    -------
    list of (conditions, leaf_stats) where
        conditions = [(feature_name, operator, threshold), ...]
        leaf_stats = {"n_samples": int, "predicted_class": int}
    """
    tree_ = tree_clf.tree_
    rules: list = []

    def recurse(node: int, path: list) -> None:
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            values = tree_.value[node][0]
            predicted_class = int(np.argmax(values))
            if predicted_class == 1:
                rules.append((
                    list(path),
                    {
                        "n_samples": int(tree_.n_node_samples[node]),
                        "predicted_class": predicted_class,
                    },
                ))
            return

        feat = feature_names[tree_.feature[node]]
        threshold = float(tree_.threshold[node])
        recurse(tree_.children_left[node],  path + [(feat, "<=", threshold)])
        recurse(tree_.children_right[node], path + [(feat, ">",  threshold)])

    recurse(0, [])
    return rules


def extract_surrogate_rules(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    threshold: float,
    depths: list | None = None,
    min_samples_leaf: int = 30,
) -> tuple:
    """Multi-depth surrogate sweep with both model-score and true-label targets.

    For each depth, two surrogates are fitted:
      (a) **Model-score surrogate**: pseudo-labels = (predict_proba >= threshold)
      (b) **True-label surrogate**: labels = y_train (captures ground-truth patterns)

    Returns
    -------
    (all_rules, best_surrogate, numeric_cols)
    """
    if depths is None:
        depths = [3, 4, 5, 6]

    probs = model.predict_proba(X_train)[:, 1]
    pseudo_labels = (probs >= threshold).astype(int)
    logger.info(
        "    Pseudo-labels: %.2f%% positive (threshold=%.4f)",
        pseudo_labels.mean() * 100, threshold,
    )

    all_rules = []
    seen_rule_texts = set()
    best_surrogate = None
    numeric_cols = None

    for depth in depths:
        for target_name, labels in [
            ("model_score", pseudo_labels),
            ("true_label", y_train.values),
        ]:
            surrogate, num_cols = build_surrogate(
                X_train, labels,
                max_depth=depth,
                min_samples_leaf=min_samples_leaf,
            )
            if numeric_cols is None:
                numeric_cols = num_cols
            if target_name == "model_score":
                best_surrogate = surrogate

            rules = _extract_leaf_rules(surrogate, num_cols)
            for conditions, leaf_stats in rules:
                rule_text = _conditions_to_text(conditions)
                if rule_text not in seen_rule_texts and rule_text != "(no conditions)":
                    seen_rule_texts.add(rule_text)
                    leaf_stats["source"] = f"surrogate_d{depth}_{target_name}"
                    all_rules.append((conditions, leaf_stats))

    logger.info(
        "    Extracted %d unique rule(s) (depths=%s, targets=model_score+true_label).",
        len(all_rules), depths,
    )
    return all_rules, best_surrogate, numeric_cols


# ── Rule scoring ───────────────────────────────────────────────────────────────

def _apply_rule(conditions: list, X: pd.DataFrame) -> pd.Series:
    """AND all conditions; returns a boolean Series aligned to X's index."""
    mask = pd.Series(True, index=X.index)
    for feat, op, val in conditions:
        if feat not in X.columns:
            continue
        col = X[feat]
        if op == ">":
            mask &= col > val
        elif op == "<=":
            mask &= col <= val
    return mask


def evaluate_rules(rules: list, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Score each rule on (X, y) and return a DataFrame ranked by lift.

    Metrics
    -------
    precision   : fraction of flagged transactions that are actually fraud.
    recall      : fraction of all fraud caught by the rule.
    f1          : harmonic mean of precision and recall.
    coverage    : fraction of all transactions the rule fires on.
    lift        : precision / base_fraud_rate.
    n_flagged   : absolute number of transactions flagged.
    n_fraud_hit : absolute number of fraud transactions caught.
    n_conditions: number of AND clauses (after simplification).
    source      : which surrogate configuration produced the rule.
    """
    fraud_rate = y.mean()
    n_fraud_total = int(y.sum())
    rows: list = []

    for conditions, metadata in rules:
        mask = _apply_rule(conditions, X)
        n_flagged = int(mask.sum())
        if n_flagged == 0:
            continue

        y_pred = mask.astype(int)
        n_fraud_hit = int((mask & (y == 1)).sum())
        prec = precision_score(y, y_pred, zero_division=0.0)
        rec  = recall_score(y, y_pred, zero_division=0.0)
        f1   = f1_score(y, y_pred, zero_division=0.0)
        cov  = n_flagged / len(X)
        lift = prec / fraud_rate if fraud_rate > 0 else 0.0

        rows.append({
            "rule":          _conditions_to_text(conditions),
            "precision":     round(prec, 4),
            "recall":        round(rec,  4),
            "f1":            round(f1,   4),
            "coverage":      round(cov,  6),
            "lift":          round(lift, 2),
            "n_flagged":     n_flagged,
            "n_fraud_hit":   n_fraud_hit,
            "n_fraud_total": n_fraud_total,
            "n_conditions":  len(_simplify_conditions(conditions)),
            "source":        metadata.get("source", "unknown"),
        })

    if not rows:
        logger.warning("  No fraud-majority rules fired on the evaluation set.")
        return pd.DataFrame()

    df = (
        pd.DataFrame(rows)
        .sort_values("lift", ascending=False)
        .drop_duplicates(subset=["rule"])
        .reset_index(drop=True)
    )
    return df


def evaluate_combined_rules(
    rules: list,
    rules_df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = 10,
) -> pd.DataFrame:
    """Evaluate the OR-combination of the top-N rules by lift.

    Returns a DataFrame with one row per cumulative rule count (1..top_n),
    showing how precision and recall change as more rules are added.
    """
    fraud_rate = y.mean()
    n_fraud_total = int(y.sum())

    rule_text_to_conditions = {}
    for conditions, _ in rules:
        text = _conditions_to_text(conditions)
        if text not in rule_text_to_conditions:
            rule_text_to_conditions[text] = conditions

    top_rules = rules_df.head(top_n)
    combined_mask = pd.Series(False, index=X.index)
    rows = []

    for i, (_, row) in enumerate(top_rules.iterrows(), 1):
        conditions = rule_text_to_conditions.get(row["rule"])
        if conditions is None:
            continue
        combined_mask |= _apply_rule(conditions, X)

        n_flagged = int(combined_mask.sum())
        n_fraud_hit = int((combined_mask & (y == 1)).sum())
        prec = n_fraud_hit / n_flagged if n_flagged > 0 else 0.0
        rec = n_fraud_hit / n_fraud_total if n_fraud_total > 0 else 0.0
        rows.append({
            "n_rules":       i,
            "last_rule":     row["rule"][:80],
            "precision":     round(prec, 4),
            "recall":        round(rec,  4),
            "n_flagged":     n_flagged,
            "n_fraud_hit":   n_fraud_hit,
            "n_fraud_total": n_fraud_total,
            "lift":          round(prec / fraud_rate, 2) if fraud_rate > 0 else 0.0,
        })

    return pd.DataFrame(rows)


# ── Visualisations ─────────────────────────────────────────────────────────────

def plot_rule_performance(rules_df: pd.DataFrame, name: str = "BestModel") -> None:
    """Precision–Recall bubble chart.

    Each bubble is one extracted rule.  Bubble size encodes coverage (fraction
    of transactions the rule fires on) and colour encodes lift.  The top-5
    rules by lift are labelled.
    """
    if rules_df.empty:
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 7))
    scatter = ax.scatter(
        rules_df["recall"],
        rules_df["precision"],
        s=rules_df["coverage"] * 3000 + 30,
        c=rules_df["lift"],
        cmap="YlOrRd",
        alpha=0.75,
        edgecolors="grey",
        linewidths=0.5,
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Lift")

    ax.set_xlabel("Recall  (fraction of all fraud caught by the rule)")
    ax.set_ylabel("Precision  (fraction of flagged transactions that are fraud)")
    ax.set_title(
        f"Extracted Fraud Rules — Precision vs Recall  [{name}]\n"
        "(bubble size ∝ coverage · colour = lift)"
    )
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    for i, row in rules_df.head(5).iterrows():
        label = row["rule"]
        if len(label) > 65:
            label = label[:62] + "…"
        ax.annotate(
            f"#{i + 1}: {label}",
            (row["recall"], row["precision"]),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=7,
        )

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"rules_performance_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", path)


def plot_rule_leaderboard(
    rules_df: pd.DataFrame,
    name: str = "BestModel",
    top_n: int = 15,
) -> None:
    """Horizontal bar chart of the top-N rules by lift.

    Each bar is annotated with the rule's precision and recall so analysts can
    quickly judge coverage versus accuracy trade-offs.
    """
    if rules_df.empty:
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    top = rules_df.head(top_n).copy()
    top["rule_short"] = top["rule"].apply(
        lambda r: r if len(r) <= 72 else r[:69] + "…"
    )

    fig, ax = plt.subplots(figsize=(14, max(4, len(top) * 0.6)))
    palette = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(top)))[::-1]
    ax.barh(range(len(top)), top["lift"].values, color=palette)

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["rule_short"].values, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Lift  (precision / base fraud rate)")
    ax.set_title(f"Top {min(top_n, len(top))} Fraud Rules by Lift  [{name}]")

    # Annotate precision / recall / fraud count at each bar end
    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(
            row["lift"] + 0.03,
            i,
            f"P={row['precision']:.0%}  R={row['recall']:.0%}  "
            f"fraud={row.get('n_fraud_hit', '?')}",
            va="center",
            fontsize=7.5,
        )

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"rules_leaderboard_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", path)


def plot_cumulative_recall(combined_df: pd.DataFrame, name: str = "BestModel") -> None:
    """Line chart showing recall growth as rules are added (OR-combination)."""
    if combined_df.empty:
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(
        combined_df["n_rules"],
        combined_df["recall"],
        marker="o",
        color="tab:blue",
        linewidth=2,
        label="Cumulative Recall",
    )
    ax1.set_xlabel("Number of Rules (OR-combined, sorted by lift)")
    ax1.set_ylabel("Recall", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim([0, 1])

    ax2 = ax1.twinx()
    ax2.plot(
        combined_df["n_rules"],
        combined_df["precision"],
        marker="s",
        color="tab:orange",
        linewidth=2,
        linestyle="--",
        label="Cumulative Precision",
    )
    ax2.set_ylabel("Precision", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_ylim([0, max(combined_df["precision"].max() * 1.2, 0.1)])

    ax1.set_title(f"Cumulative Recall & Precision as Rules Added  [{name}]")
    ax1.set_xticks(combined_df["n_rules"].values)

    for _, row in combined_df.iterrows():
        ax1.annotate(
            f"{row['n_fraud_hit']}/{row['n_fraud_total']}",
            (row["n_rules"], row["recall"]),
            textcoords="offset points",
            xytext=(0, 10),
            fontsize=7,
            ha="center",
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"rules_cumulative_recall_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", path)


# ── Orchestrator ───────────────────────────────────────────────────────────────

def run_rule_extraction(
    best_model,
    best_model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    recall_threshold: float = 0.50,
    surrogate_depths: list | None = None,
    all_models: dict | None = None,
) -> pd.DataFrame:
    """Full rule extraction pipeline with multi-depth sweep and combined coverage.

    Parameters
    ----------
    best_model        : fitted sklearn Pipeline (possibly calibrated).
    best_model_name   : str — for file names and titles.
    X_train           : pd.DataFrame — raw training features.
    y_train           : pd.Series — training labels.
    X_val             : pd.DataFrame — raw validation features.
    y_val             : pd.Series — validation labels.
    recall_threshold  : float — operating-point threshold.
    surrogate_depths  : list — depth values to sweep (default [3, 4, 5, 6]).
    all_models        : dict — {name: model} for multi-model extraction.

    Returns
    -------
    pd.DataFrame of scored rules sorted by lift descending.
    """
    if surrogate_depths is None:
        surrogate_depths = [3, 4, 5, 6]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Determine which models to extract from ─────────────────────────
    models_to_extract = {best_model_name: best_model}
    if all_models:
        for name, model in all_models.items():
            if name not in models_to_extract:
                models_to_extract[name] = model

    all_candidate_rules = []
    best_surrogate_for_export = None
    numeric_cols = None

    for model_name, model in models_to_extract.items():
        logger.info(
            "  Extracting surrogate rules from %s (depths=%s, threshold=%.4f) …",
            model_name, surrogate_depths, recall_threshold,
        )
        rules, surrogate, num_cols = extract_surrogate_rules(
            model, X_train, y_train,
            threshold=recall_threshold,
            depths=surrogate_depths,
        )
        for conditions, metadata in rules:
            metadata["model"] = model_name
        all_candidate_rules.extend(rules)

        if model_name == best_model_name:
            best_surrogate_for_export = surrogate
            numeric_cols = num_cols

    logger.info(
        "  Total candidate rules across all models: %d",
        len(all_candidate_rules),
    )

    # ── Persist the best surrogate tree for manual inspection ──────────
    if best_surrogate_for_export is not None and numeric_cols is not None:
        tree_text = export_text(
            best_surrogate_for_export,
            feature_names=numeric_cols,
            max_depth=10,
        )
        tree_path = os.path.join(OUTPUT_DIR, "surrogate_tree.txt")
        with open(tree_path, "w", encoding="utf-8") as fh:
            fh.write(tree_text)
        logger.info("  Surrogate tree structure saved to %s", tree_path)

    if not all_candidate_rules:
        logger.warning(
            "  No fraud-majority leaves found. "
            "Try increasing surrogate_depths or lowering recall_threshold."
        )
        return pd.DataFrame()

    # ── Evaluate rules on validation set ──────────────────────────────
    eval_cols = numeric_cols or X_val.select_dtypes(include=[np.number]).columns.tolist()
    logger.info(
        "  Scoring %d candidate rule(s) on validation set (%d rows, %d fraud) …",
        len(all_candidate_rules), len(X_val), int(y_val.sum()),
    )
    rules_df = evaluate_rules(all_candidate_rules, X_val[eval_cols], y_val)

    if rules_df.empty:
        logger.warning("  No rules fired on the validation set.")
        return rules_df

    # ── Persist scored rules ──────────────────────────────────────────
    rules_path = os.path.join(OUTPUT_DIR, "rules_summary.csv")
    rules_df.to_csv(rules_path, index=False)
    logger.info(
        "  Rules summary saved to %s  (%d unique rules)",
        rules_path, len(rules_df),
    )

    display_cols = ["rule", "precision", "recall", "n_fraud_hit", "lift", "source"]
    available_cols = [c for c in display_cols if c in rules_df.columns]
    logger.info(
        "\n  Top 10 rules by lift:\n%s",
        rules_df.head(10)[available_cols].to_string(index=False),
    )

    # ── Plots ────────────────────────────────────────────────────────
    plot_rule_performance(rules_df, name=best_model_name)
    plot_rule_leaderboard(rules_df, name=best_model_name)

    # ── Combined OR-coverage analysis ────────────────────────────────
    combined_df = evaluate_combined_rules(
        all_candidate_rules, rules_df,
        X_val[eval_cols], y_val,
        top_n=min(15, len(rules_df)),
    )
    if not combined_df.empty:
        combined_path = os.path.join(OUTPUT_DIR, "rules_combined_coverage.csv")
        combined_df.to_csv(combined_path, index=False)
        logger.info("  Combined OR-coverage saved to %s", combined_path)
        logger.info(
            "\n  Combined OR-coverage (top rules):\n%s",
            combined_df.to_string(index=False),
        )
        plot_cumulative_recall(combined_df, name=best_model_name)

    return rules_df
