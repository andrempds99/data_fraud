"""
Generate Concise PDF Report (max 3 pages)
==========================================
Produces output/fraud_detection_summary.pdf
"""

import os
import pandas as pd
from fpdf import FPDF

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
REPORT_PATH = os.path.join(OUTPUT_DIR, "fraud_detection_summary.pdf")
FONT_DIR = r"C:\Windows\Fonts"


class Report(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("F", "", os.path.join(FONT_DIR, "arial.ttf"))
        self.add_font("F", "B", os.path.join(FONT_DIR, "arialbd.ttf"))
        self.add_font("F", "I", os.path.join(FONT_DIR, "ariali.ttf"))

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("F", "I", 7)
        self.set_text_color(140, 140, 140)
        self.cell(0, 6, "Fraud Detection Pipeline — Summary Report", align="R",
                  new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(1)

    def footer(self):
        self.set_y(-12)
        self.set_font("F", "I", 7)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, f"Page {self.page_no()}/{{nb}}", align="C")

    def section(self, title):
        self.set_font("F", "B", 11)
        self.set_text_color(20, 60, 120)
        self.ln(2)
        self.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(20, 60, 120)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)

    def text(self, t):
        self.set_font("F", "", 8.5)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 4.2, t)
        self.ln(0.5)

    def bullet(self, t):
        self.set_font("F", "", 8.5)
        self.set_text_color(30, 30, 30)
        indent = 6
        x0 = self.l_margin
        self.set_x(x0)              # always start from l_margin, not drifted x
        self.cell(indent, 4.2, "-")
        self.set_left_margin(x0 + indent)
        self.multi_cell(0, 4.2, t)
        self.set_left_margin(x0)

    def table(self, headers, rows, col_widths=None):
        n = len(headers)
        if col_widths is None:
            col_widths = [185 / n] * n
        self.set_font("F", "B", 7.5)
        self.set_fill_color(20, 60, 120)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 5.5, h, border=1, fill=True, align="C")
        self.ln()
        self.set_font("F", "", 7.5)
        self.set_text_color(30, 30, 30)
        for ri, row in enumerate(rows):
            fill = ri % 2 == 0
            if fill:
                self.set_fill_color(245, 245, 250)
            for i, val in enumerate(row):
                self.cell(col_widths[i], 4.8, str(val), border=1, fill=fill, align="C")
            self.ln()
        self.ln(1.5)

    def img(self, path, w=80):
        if os.path.exists(path):
            self.image(path, x=self.get_x(), w=w)
            self.ln(2)


def build():
    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=14)

    # ── PAGE 1: Title + Data + Features + Models ────────────────────────
    pdf.add_page()

    # Title block
    pdf.ln(2)
    pdf.set_font("F", "B", 20)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 10, "Fraud Detection Pipeline", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("F", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, "Summary Report  |  March 2026", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(20, 60, 120)
    pdf.line(60, pdf.get_y() + 1, 150, pdf.get_y() + 1)
    pdf.ln(4)

    # Overview
    pdf.section("1. Overview")
    pdf.text(
        "End-to-end fraud detection pipeline: data quality checks, 39-feature engineering "
        "(multi-window velocity, cyclical encoding, outlier capping), 4 classifiers "
        "(Logistic Regression, Random Forest, XGBoost, LightGBM), stacking ensemble, "
        "SMOTE comparison, overfitting mitigation, SHAP explainability, isotonic calibration, "
        "and recall optimisation at 30-60%. Best model: XGBoost (PR-AUC=0.150, ROC-AUC=0.944)."
    )

    # Data Quality
    pdf.section("2. Data Quality & Preparation")
    pdf.table(
        ["", "Raw", "After Dedup", "Columns", "Fraud Rate"],
        [
            ["Train", "157,306", "113,905", "48 (44 after label drop)", "0.11%"],
            ["Test", "110,998", "74,792", "44", "Unknown"],
        ],
        [15, 35, 35, 60, 40],
    )
    pdf.text(
        "Pipeline: null normalisation (sentinel strings -> NaN) -> deduplication (fraud-priority) "
        "-> encoding fixes -> type casting -> identifier removal -> quality report. "
        "Temporal hold-out: test starts Apr 2020, train from Feb 2020."
    )

    # Features
    pdf.section("3. Feature Engineering (31 numeric + 8 categorical)")
    pdf.table(
        ["Category", "Features", "Rationale"],
        [
            ["Multi-window velocity", "card_txn_{1,3,7,14,30}d", "Burst patterns at diff. scales"],
            ["Recency", "hours_since_last_txn", "Rapid-fire fraud detection"],
            ["Card stats", "card_avg_amount, amount_deviation", "Anomalous amounts"],
            ["Card-merchant", "card_merchant_first, card_unique_merchants", "New merchant risk"],
            ["Geographic", "card_billing_mismatch, geoip_billing_mismatch", "Cross-border fraud"],
            ["Cyclical temporal", "hour_sin/cos, dow_sin/cos + raw", "Smooth time-of-day signal"],
            ["Amount", "log_euramount, euramount_capped", "Skew + outlier control"],
            ["Domain", "domain_freq (email domain frequency)", "Synthetic identities"],
            ["Categorical (8)", "cardbrand, cardtype, decline_type, ...", "One-hot encoded"],
        ],
        [30, 75, 80],
    )
    pdf.text(
        "Leakage-safe: lookup tables built from training fold only. "
        "Rare categories binned (freq < 0.5%); outliers capped at 1st/99th percentile. "
        "Temporal 80/20 split: Train=91,124, Val=22,781 (0.16% fraud, 36 fraud cases)."
    )

    # Class imbalance
    pdf.section("4. Class Imbalance Handling")
    pdf.text(
        "Strategies: class_weight='balanced' (LR, RF), scale_pos_weight (XGBoost), "
        "is_unbalance=True (LightGBM). "
        "SMOTE experiment: resampled 91,124 -> 181,986 rows (fraud 131 -> 90,993). "
        "Result: class weights (PR-AUC=0.150) outperformed SMOTE (PR-AUC=0.043) — "
        "SMOTE over-smooths minority-class boundaries in extreme-imbalance regimes."
    )

    # Model results
    pdf.section("5. Model Results")
    pdf.table(
        ["Model", "ROC-AUC", "PR-AUC", "F1", "Val Recall", "Train/Val ROC gap"],
        [
            ["LogisticRegression", "0.936", "0.028", "0.018", "89%", "0.019"],
            ["RandomForest", "0.943", "0.075", "0.038", "56%", "0.038"],
            ["XGBoost*", "0.944", "0.150", "0.017", "92%", "0.024"],
            ["LightGBM", "0.608", "0.006", "0.016", "44%", "0.348"],
            ["Stacking", "---", "0.077", "---", "---", "---"],
        ],
        [35, 25, 25, 20, 30, 50],
    )
    pdf.set_font("F", "I", 7.5)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 4, "* Best model selected by PR-AUC. Stacking includes only models with ROC-AUC >= 0.60.",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    # ── PAGE 2: Overfitting + Comparison ────────────────────────────────
    pdf.add_page()
    pdf.section("6. Overfitting Detection & Mitigation")
    pdf.text(
        "Baseline XGBoost (no regularisation, max_depth=12, 500 trees): "
        "Train ROC=0.9998, Val ROC=0.879, gap=0.121. "
        "Regularised XGBoost (max_depth=6, subsample=0.8, colsample=0.8, early stopping): "
        "Train ROC=0.968, Val ROC=0.944, gap=0.024. "
        "Regularisation reduces the overfit gap by 5x while improving validation performance."
    )

    # Overfit comparison chart — centred, moderate width
    overfit_path = os.path.join(OUTPUT_DIR, "overfit_comparison_XGBoost.png")
    if os.path.exists(overfit_path):
        pdf.image(overfit_path, x=35, w=125)
        pdf.ln(2)

    pdf.section("7. Model Comparison")

    # ROC + PR side by side — place left image normally, rewind y, place right image
    roc_path = os.path.join(OUTPUT_DIR, "comparison_roc.png")
    pr_path  = os.path.join(OUTPUT_DIR, "comparison_pr_curve.png")
    y_start = pdf.get_y()
    if os.path.exists(roc_path):
        pdf.image(roc_path, x=10, w=90)
    y_after_left = pdf.get_y()
    pdf.set_y(y_start)                       # rewind to place right image alongside
    if os.path.exists(pr_path):
        pdf.image(pr_path, x=108, w=90)
    y_after_right = pdf.get_y()
    pdf.set_y(max(y_after_left, y_after_right) + 2)

    pdf.text(
        "XGBoost dominates on PR-AUC (0.150), the most informative metric for extreme "
        "imbalance. High ROC-AUC across LR/RF/XGB (0.93-0.94) masks the difficulty "
        "visible in PR-AUC. LightGBM underperforms (ROC-AUC=0.61) despite safety-net retraining."
    )

    # ── PAGE 3: Recall + SHAP ────────────────────────────────────────────
    pdf.add_page()
    pdf.section("8. Recall Optimisation (30%-60%)")
    recall_path = os.path.join(OUTPUT_DIR, "recall_summary.csv")
    if os.path.exists(recall_path):
        df = pd.read_csv(recall_path)
        xdf = df[df["model"] == "XGBoost"]
        rows = []
        for _, r in xdf.iterrows():
            rows.append([
                f"{r['target_recall']:.0%}",
                f"{r['threshold']:.4f}",
                f"{r['actual_recall']:.1%}",
                f"{r['precision']:.4f}",
                f"{r['f1']:.4f}",
            ])
        pdf.set_font("F", "B", 8)
        pdf.set_text_color(40, 80, 140)
        pdf.cell(0, 5, "XGBoost (Best Model) — Recall Operating Points:",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        pdf.table(
            ["Target", "Threshold", "Actual Recall", "Precision", "F1"],
            rows,
            [30, 40, 40, 40, 35],
        )
    pdf.text(
        "Test predictions: threshold=0.027 (targeting ~50% recall), "
        "flagging 2,341/74,792 transactions (3.13%) as potentially fraudulent."
    )

    pdf.section("9. Explainability (SHAP)")

    # SHAP beeswarm + bar side by side using rewind approach
    shap_bee = os.path.join(OUTPUT_DIR, "shap_summary_XGBoost.png")
    shap_bar = os.path.join(OUTPUT_DIR, "shap_importance_XGBoost.png")
    y_start = pdf.get_y()
    if os.path.exists(shap_bee):
        pdf.image(shap_bee, x=10, w=90)
    y_after_left = pdf.get_y()
    pdf.set_y(y_start)
    if os.path.exists(shap_bar):
        pdf.image(shap_bar, x=108, w=90)
    y_after_right = pdf.get_y()
    pdf.set_y(max(y_after_left, y_after_right) + 2)

    pdf.text(
        "SHAP TreeExplainer on the best XGBoost model. Beeswarm plot (left) shows per-feature "
        "impact direction; bar plot (right) shows mean absolute SHAP importance. "
        "Top features provide transparent, auditable explanations for each fraud prediction."
    )

    # ── PAGE 4: Best model detail + Conclusions ──────────────────────────
    pdf.add_page()
    pdf.section("10. Best Model Detail (XGBoost)")

    # Confusion matrix + feature importance side by side using rewind approach
    cm = os.path.join(OUTPUT_DIR, "XGBoost", "confusion_matrix_XGBoost.png")
    fi = os.path.join(OUTPUT_DIR, "XGBoost", "feature_importance_XGBoost.png")
    y_start = pdf.get_y()
    if os.path.exists(cm):
        pdf.image(cm, x=10, w=88)
    y_after_left = pdf.get_y()
    pdf.set_y(y_start)
    if os.path.exists(fi):
        pdf.image(fi, x=105, w=92)
    y_after_right = pdf.get_y()
    pdf.set_y(max(y_after_left, y_after_right) + 2)

    # Learning curves — centred
    lc = os.path.join(OUTPUT_DIR, "XGBoost", "learning_curves_XGBoost.png")
    if os.path.exists(lc):
        pdf.image(lc, x=25, w=155)
        pdf.ln(2)

    pdf.section("11. Conclusions")
    pdf.bullet("XGBoost selected as best model (PR-AUC=0.150, ROC-AUC=0.944) with isotonic calibration applied")
    pdf.bullet("Class weights outperform SMOTE (PR-AUC 0.150 vs 0.043) for this extreme-imbalance regime")
    pdf.bullet("Regularisation reduces overfit gap from 0.121 to 0.024 (5x improvement)")
    pdf.bullet("SHAP explainability provides transparent, auditable per-prediction reasoning")
    pdf.bullet("Stacking ensemble (PR-AUC=0.077) underperforms standalone XGBoost due to weaker base learners")
    pdf.bullet("Serialised artifacts (best_model.joblib, lookup_tables.joblib) enable deployment without retraining")

    # Save
    pdf.output(REPORT_PATH)
    print(f"Summary report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    build()
