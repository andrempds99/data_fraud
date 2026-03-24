"""
Generate PDF Report for Fraud Detection Pipeline
=================================================
Produces output/fraud_detection_report.pdf
"""

import os
import pandas as pd
from fpdf import FPDF

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
REPORT_PATH = os.path.join(OUTPUT_DIR, "fraud_detection_report.pdf")

MODELS = ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM"]

FONT_DIR = r"C:\Windows\Fonts"


class Report(FPDF):
    def __init__(self):
        super().__init__()
        # Register Unicode-capable TTF fonts
        self.add_font("CustomFont", "", os.path.join(FONT_DIR, "arial.ttf"))
        self.add_font("CustomFont", "B", os.path.join(FONT_DIR, "arialbd.ttf"))
        self.add_font("CustomFont", "I", os.path.join(FONT_DIR, "ariali.ttf"))
        self.add_font("CustomFont", "BI", os.path.join(FONT_DIR, "arialbi.ttf"))
        self.add_font("CustomMono", "", os.path.join(FONT_DIR, "consola.ttf"))

    def header(self):
        self.set_font("CustomFont", "B", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "Fraud Detection Pipeline -- Technical Report", align="R", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("CustomFont", "I", 8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def chapter_title(self, title, level=1):
        if level == 1:
            self.set_font("CustomFont", "B", 16)
            self.set_text_color(20, 60, 120)
            self.ln(4)
            self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(20, 60, 120)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)
        elif level == 2:
            self.set_font("CustomFont", "B", 13)
            self.set_text_color(40, 80, 140)
            self.ln(3)
            self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
            self.ln(2)
        else:
            self.set_font("CustomFont", "B", 11)
            self.set_text_color(60, 60, 60)
            self.ln(2)
            self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
            self.ln(1)

    def body_text(self, text):
        self.set_font("CustomFont", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bullet(self, text):
        self.set_font("CustomFont", "", 10)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        self.cell(6, 5.5, "  -")
        self.multi_cell(170, 5.5, text)

    def code_block(self, text):
        self.set_font("CustomMono", "", 8)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        self.multi_cell(0, 4.5, text, fill=True)
        self.ln(2)

    def add_image_safe(self, path, w=170):
        if os.path.exists(path):
            if self.get_y() + 90 > 270:
                self.add_page()
            self.image(path, x=20, w=w)
            self.ln(4)
        else:
            self.body_text(f"[Image not found: {os.path.basename(path)}]")

    def metric_table(self, headers, rows):
        self.set_font("CustomFont", "B", 9)
        self.set_fill_color(20, 60, 120)
        self.set_text_color(255, 255, 255)
        col_w = 180 / len(headers)
        for h in headers:
            self.cell(col_w, 7, h, border=1, fill=True, align="C")
        self.ln()
        self.set_font("CustomFont", "", 9)
        self.set_text_color(30, 30, 30)
        for i, row in enumerate(rows):
            fill = i % 2 == 0
            if fill:
                self.set_fill_color(245, 245, 250)
            for val in row:
                self.cell(col_w, 6, str(val), border=1, fill=fill, align="C")
            self.ln()
        self.ln(3)


def build_report():
    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # -- TITLE PAGE ------------------------------------------------------
    pdf.add_page()
    pdf.ln(50)
    pdf.set_font("CustomFont", "B", 28)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 15, "Fraud Detection Pipeline", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("CustomFont", "", 16)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, "Technical Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_draw_color(20, 60, 120)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(10)
    pdf.set_font("CustomFont", "", 11)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 8, "Data Science Take-Home Assessment", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "March 2026", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(30)

    pdf.set_font("CustomFont", "B", 11)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 8, "Contents", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("CustomFont", "", 10)
    pdf.set_text_color(30, 30, 30)
    toc = [
        "1. Executive Summary",
        "2. Data Quality Assessment",
        "3. Feature Engineering",
        "4. Class Imbalance Handling",
        "5. Model Training & Architecture",
        "6. Evaluation Metrics & Overfitting Analysis",
        "7. Model Comparison",
        "8. Recall Optimisation (30%\u201360%)",
        "9. Test Predictions",
        "10. Conclusions & Recommendations",
    ]
    for item in toc:
        pdf.cell(0, 6, f"    {item}", new_x="LMARGIN", new_y="NEXT")

    # -- 1. EXECUTIVE SUMMARY --------------------------------------------
    pdf.add_page()
    pdf.chapter_title("1. Executive Summary")
    pdf.body_text(
        "This report documents an end-to-end fraud detection pipeline built for the Fraudio "
        "data science take-home assessment. The pipeline loads transactional payment data, "
        "performs data quality checks, engineers fraud-relevant features, trains four classifiers, "
        "evaluates them with a focus on precision-recall trade-offs, and optimises for target "
        "recall levels between 30% and 60%."
    )
    pdf.body_text(
        "Key results:"
    )
    pdf.bullet("Dataset: 157,306 training transactions (0.11% fraud rate after label extraction), 110,998 test transactions")
    pdf.bullet("After deduplication: 113,905 train rows, 74,792 test rows")
    pdf.bullet("27 engineered features (20 numeric + 8 categorical, processed via ColumnTransformer)")
    pdf.bullet("4 models trained: Logistic Regression, Random Forest, XGBoost, LightGBM")
    pdf.bullet("Best model: XGBoost (PR-AUC = 0.051, ROC-AUC = 0.932)")
    pdf.bullet("Temporal train/validation split (80/20) to simulate production conditions")
    pdf.bullet("Test predictions generated at ~50% recall operating point (threshold = 0.025)")

    # -- 2. DATA QUALITY -------------------------------------------------
    pdf.add_page()
    pdf.chapter_title("2. Data Quality Assessment")

    pdf.chapter_title("2.1 Raw Data Overview", level=2)
    pdf.metric_table(
        ["Property", "Train", "Test"],
        [
            ["Rows (raw)", "157,306", "110,998"],
            ["Columns", "48 (incl. 4 labels)", "44"],
            ["Fraud rate", "0.11%", "Unknown"],
            ["Date range", "Feb 2020+", "Apr 2020+"],
        ],
    )
    pdf.body_text(
        "The test set is a temporal hold-out starting April 2020, with 4 label columns "
        "(cb_fraudflag, chargeback_bank_date, chargeback_reason_code, fraudimportdate) "
        "removed -- consistent with a real-world deployment scenario."
    )

    pdf.chapter_title("2.2 Quality Pipeline (6 Steps)", level=2)
    pdf.body_text("The data quality pipeline applies the following transformations in order:")

    pdf.chapter_title("Step 1: Null Normalisation", level=3)
    pdf.body_text(
        'Sentinel strings ("none", "n/a", "na", "nan", "null", "") are replaced with np.nan '
        "across all object columns after lowercasing and stripping whitespace. This ensures "
        "consistent missing-value handling downstream."
    )

    pdf.chapter_title("Step 2: Deduplication", level=3)
    pdf.body_text(
        "Duplicate transactionid rows are removed. The deduplication strategy prioritises: "
        "(1) fraud-positive rows first (so chargebacks are never silently discarded), then "
        '(2) transaction-type priority (refund > auth_capture > capture > auth). '
        "This reduced train from 157,306 to 113,905 rows and test from 110,998 to 74,792 rows."
    )

    pdf.chapter_title("Step 3: Response Code Encoding Fixes", level=3)
    pdf.body_text(
        'Garbled UTF-8 encoded response codes (e.g. "limitgjberschritten,doch-funktionmg6glich") '
        "are mapped to clean labels (e.g. \"limit_exceeded_possible\")."
    )

    pdf.chapter_title("Step 4: Type Casting", level=3)
    pdf.body_text(
        "Boolean string columns (cvvused, recurring, threedsused, etc.) are cast to Int8. "
        "Numeric fields (cardbin, euramount) are coerced. ISO 8601 timestamps are parsed to "
        "datetime with UTC awareness."
    )

    pdf.chapter_title("Step 5: Drop Identifiers", level=3)
    pdf.body_text(
        "High-cardinality identifiers with no predictive signal are dropped: transactionid, "
        "transactionip, approval_code, submerchant, merchanturl. Note that saltedhash is "
        "retained at this stage because it is needed for velocity feature engineering."
    )

    pdf.chapter_title("Step 6: Quality Report", level=3)
    pdf.body_text(
        "A comprehensive quality report is logged showing missing-value percentages and "
        "cardinality per column. Key findings from the cleaned training set:"
    )
    pdf.metric_table(
        ["Column", "Missing %", "Notes"],
        [
            ["merchantcountry", "100.0%", "Entirely null -- dropped in FE"],
            ["cardholder_disposabledomain", "99.0%", "Nearly all null"],
            ["decline_type", "57.6%", "Missing when txn approved"],
            ["State", "56.3%", "Geographic data gap"],
            ["City", "22.8%", "Geographic data gap"],
            ["geoip_country_code", "4.7%", "Minor gaps"],
            ["issuingbank", "2.0%", "Minor gaps"],
        ],
    )

    # -- 3. FEATURE ENGINEERING ------------------------------------------
    pdf.add_page()
    pdf.chapter_title("3. Feature Engineering")
    pdf.body_text(
        "Feature engineering is applied after the temporal train/validation split to prevent "
        "data leakage. Lookup tables are built from the training fold only -- validation and test "
        "rows never influence training statistics."
    )

    pdf.chapter_title("3.1 Timestamp Features", level=2)
    pdf.body_text(
        "From authtimestamp: hour (0-23), dayofweek (0-6), is_weekend (binary), day (1-31). "
        "These capture temporal fraud patterns (e.g. higher fraud rates during certain hours)."
    )

    pdf.chapter_title("3.2 Country Mismatch Flags", level=2)
    pdf.body_text(
        "Two binary flags engineered to detect geographic inconsistencies: "
        "card_billing_mismatch (card issuing country != billing country) and "
        "geoip_billing_mismatch (IP geolocation country != billing country). "
        "Cross-border mismatches are strong fraud signals."
    )

    pdf.chapter_title("3.3 Velocity & Aggregation Features", level=2)
    pdf.body_text(
        "card_txn_7d: Rolling 7-day transaction count per card, computed via binary search on "
        "sorted timestamp history. This captures burst-like fraud patterns without lifetime bias."
    )
    pdf.body_text(
        "card_avg_amount: Lifetime average transaction amount per card. "
        "amount_deviation: Current transaction amount minus card average -- flags abnormally "
        "large or small transactions for a given cardholder."
    )
    pdf.body_text(
        "merchant_txn_count: Total transaction count per merchant -- proxies merchant "
        "popularity / legitimacy."
    )

    pdf.chapter_title("3.4 Domain Features", level=2)
    pdf.body_text(
        "domain_freq: Frequency encoding of cardholder email domain. Rare or disposable "
        "domains may indicate synthetic identities."
    )

    pdf.chapter_title("3.5 Amount Features", level=2)
    pdf.body_text(
        "log_euramount: Log1p transformation of euramount to reduce skewness and improve "
        "model performance on right-skewed transaction amounts."
    )

    pdf.chapter_title("3.6 Final Feature Set (27 features)", level=2)
    pdf.metric_table(
        ["Type", "Count", "Features"],
        [
            ["Numeric", "20",
             "euramount, log_euramount, card_txn_7d, card_avg_amount, "
             "merchant_txn_count, amount_deviation, hour, dayofweek, day, "
             "is_weekend, card/geoip mismatches, domain_freq, booleans"],
            ["Categorical", "8",
             "cardbrand, cardtype, transactiontype, channel, "
             "terminaltype, currencyname, brandcardtype, decline_type"],
        ],
    )
    pdf.body_text(
        "Preprocessing: numeric features are median-imputed and standard-scaled. "
        "Categorical features are constant-imputed ('missing') and one-hot encoded."
    )

    # -- 4. CLASS IMBALANCE ----------------------------------------------
    pdf.add_page()
    pdf.chapter_title("4. Class Imbalance Handling")
    pdf.body_text(
        "The dataset exhibits extreme class imbalance with a fraud rate of approximately "
        "0.11-0.16% (36 fraud cases in the validation set of 22,781 rows). This ratio "
        "of ~630:1 makes standard accuracy meaningless and requires specialised strategies."
    )

    pdf.chapter_title("4.1 Strategies Employed", level=2)
    pdf.bullet("Logistic Regression & Random Forest: class_weight='balanced' -- automatically "
               "adjust weights inversely proportional to class frequencies.")
    pdf.bullet("XGBoost: scale_pos_weight = neg_count / pos_count -- explicit positive class "
               "upweighting computed from the training data.")
    pdf.bullet("LightGBM: is_unbalance=True -- built-in handling that adjusts the loss function "
               "to penalise false negatives more heavily.")
    pdf.ln(2)

    pdf.chapter_title("4.2 Evaluation Metric Choice", level=2)
    pdf.body_text(
        "Given the extreme imbalance, we prioritise PR-AUC (Average Precision) over ROC-AUC "
        "as the primary selection metric. PR-AUC is more informative when the positive class "
        "is rare because it focuses on the precision-recall trade-off rather than the "
        "true-negative-dominated ROC space."
    )

    # -- 5. MODEL TRAINING ----------------------------------------------
    pdf.add_page()
    pdf.chapter_title("5. Model Training & Architecture")

    pdf.chapter_title("5.1 Temporal Train/Validation Split", level=2)
    pdf.body_text(
        "A temporal split (80/20) is used instead of random stratified splitting. Data is "
        "sorted by authtimestamp; the earliest 80% forms the training set and the most recent "
        "20% forms validation. This mirrors production deployment where models predict on "
        "future transactions. Train: 91,124 rows, Val: 22,781 rows (0.16% fraud)."
    )

    pdf.chapter_title("5.2 Model Specifications", level=2)

    pdf.chapter_title("Logistic Regression", level=3)
    pdf.code_block(
        "class_weight='balanced', max_iter=1000, random_state=42\n"
        "Preprocessor: StandardScaler + OneHotEncoder"
    )

    pdf.chapter_title("Random Forest", level=3)
    pdf.code_block(
        "n_estimators=300, max_depth=15, min_samples_leaf=20\n"
        "class_weight='balanced', n_jobs=-1, random_state=42"
    )

    pdf.chapter_title("XGBoost", level=3)
    pdf.code_block(
        "n_estimators=500, max_depth=6, learning_rate=0.05\n"
        "scale_pos_weight=auto, subsample=0.8, colsample_bytree=0.8\n"
        "eval_metric='aucpr', random_state=42"
    )

    pdf.chapter_title("LightGBM", level=3)
    pdf.code_block(
        "n_estimators=500, max_depth=6, learning_rate=0.05\n"
        "is_unbalance=True, subsample=0.8, colsample_bytree=0.8\n"
        "random_state=42, verbose=-1"
    )

    # -- 6. EVALUATION & OVERFITTING -------------------------------------
    pdf.add_page()
    pdf.chapter_title("6. Evaluation Metrics & Overfitting Analysis")

    pdf.chapter_title("6.1 Validation Metrics Summary", level=2)
    pdf.metric_table(
        ["Model", "ROC-AUC", "PR-AUC", "F1", "Precision", "Recall"],
        [
            ["LogisticRegression", "0.9315", "0.0216", "0.013", "0.01", "0.92"],
            ["RandomForest", "0.9291", "0.0115", "0.021", "0.01", "0.75"],
            ["XGBoost", "0.9320", "0.0508", "0.071", "0.04", "0.25"],
            ["LightGBM", "0.8118", "0.0065", "0.016", "0.01", "0.78"],
        ],
    )
    pdf.body_text(
        "XGBoost achieves the highest PR-AUC (0.051) -- the most relevant metric given "
        "the extreme imbalance. While Logistic Regression achieves the highest recall (0.92), "
        "it does so at very low precision (0.01), producing excessive false positives."
    )

    pdf.chapter_title("6.2 Overfitting Analysis", level=2)
    pdf.metric_table(
        ["Model", "Train ROC", "Val ROC", "Train PR", "Val PR", "Gap PR"],
        [
            ["LogisticRegression", "0.9491", "0.9315", "0.0797", "0.0216", "0.058"],
            ["RandomForest", "0.9780", "0.9291", "0.1733", "0.0115", "0.162"],
            ["XGBoost", "0.9997", "0.9320", "0.8621", "0.0508", "0.811"],
            ["LightGBM", "0.8587", "0.8118", "0.0075", "0.0065", "0.001"],
        ],
    )
    pdf.body_text(
        "XGBoost shows significant overfitting (train PR-AUC 0.86 vs val 0.05), typical of "
        "gradient boosting on highly imbalanced data with 500 estimators. However, it still "
        "achieves the best validation PR-AUC. Potential mitigations:"
    )
    pdf.bullet("Reduce n_estimators or add early stopping on validation PR-AUC")
    pdf.bullet("Increase regularisation (lower max_depth, higher min_child_weight)")
    pdf.bullet("Use Optuna hyperparameter tuning (available via TUNE_HYPERPARAMS=1)")
    pdf.ln(2)
    pdf.body_text(
        "LightGBM shows the smallest train/val gap (0.001), but its absolute performance is "
        "the weakest. Logistic Regression is a reasonable middle ground with moderate overfitting "
        "and competitive ROC-AUC."
    )

    # Per-model plots
    for model in MODELS:
        pdf.add_page()
        pdf.chapter_title(f"6.3 {model} -- Detailed Plots", level=2)

        cm_path = os.path.join(OUTPUT_DIR, model, f"confusion_matrix_{model}.png")
        pdf.chapter_title("Confusion Matrix", level=3)
        pdf.add_image_safe(cm_path, w=130)

        roc_path = os.path.join(OUTPUT_DIR, model, f"roc_{model}.png")
        pdf.chapter_title("ROC Curve", level=3)
        pdf.add_image_safe(roc_path, w=150)

        pr_path = os.path.join(OUTPUT_DIR, model, f"pr_curve_{model}.png")
        pdf.chapter_title("Precision-Recall Curve", level=3)
        pdf.add_image_safe(pr_path, w=150)

        fi_path = os.path.join(OUTPUT_DIR, model, f"feature_importance_{model}.png")
        if os.path.exists(fi_path):
            pdf.chapter_title("Feature Importance", level=3)
            pdf.add_image_safe(fi_path, w=150)

        lc_path = os.path.join(OUTPUT_DIR, model, f"learning_curves_{model}.png")
        if os.path.exists(lc_path):
            pdf.chapter_title("Learning Curves", level=3)
            pdf.add_image_safe(lc_path, w=150)

        rop_path = os.path.join(OUTPUT_DIR, model, f"recall_operating_points_{model}.png")
        pdf.chapter_title("Recall Operating Points", level=3)
        pdf.add_image_safe(rop_path, w=150)

    # -- 7. MODEL COMPARISON ---------------------------------------------
    pdf.add_page()
    pdf.chapter_title("7. Model Comparison")
    pdf.body_text(
        "The following plots compare all four models side-by-side on the validation set."
    )

    pdf.chapter_title("7.1 ROC Comparison", level=2)
    pdf.add_image_safe(os.path.join(OUTPUT_DIR, "comparison_roc.png"), w=160)

    pdf.chapter_title("7.2 Precision-Recall Comparison", level=2)
    pdf.add_image_safe(os.path.join(OUTPUT_DIR, "comparison_pr_curve.png"), w=160)

    pdf.chapter_title("7.3 Metrics Bar Chart", level=2)
    pdf.add_image_safe(os.path.join(OUTPUT_DIR, "comparison_metrics.png"), w=170)

    pdf.body_text(
        "Key observation: All models achieve high ROC-AUC (0.81-0.93), but PR-AUC values "
        "are very low (0.006-0.051), reflecting the extreme difficulty of achieving high "
        "precision on a 0.16% fraud-rate dataset. XGBoost leads by a meaningful margin in "
        "PR-AUC -- the metric most relevant for fraud detection deployment."
    )

    # -- 8. RECALL OPTIMISATION ------------------------------------------
    pdf.add_page()
    pdf.chapter_title("8. Recall Optimisation (30%\u201360%)")
    pdf.body_text(
        "For each model, we compute the decision threshold that achieves target recall levels "
        "from 30% to 60%. This allows operational flexibility -- e.g. a higher recall catches "
        "more fraud but generates more false positives for manual review."
    )

    # Load recall summary
    recall_path = os.path.join(OUTPUT_DIR, "recall_summary.csv")
    if os.path.exists(recall_path):
        df = pd.read_csv(recall_path)
        for model in MODELS:
            mdf = df[df["model"] == model]
            if mdf.empty:
                continue
            pdf.chapter_title(f"8.1 {model}", level=2)
            rows = []
            for _, r in mdf.iterrows():
                rows.append([
                    f"{r['target_recall']:.0%}",
                    f"{r['threshold']:.4f}",
                    f"{r['actual_recall']:.2%}",
                    f"{r['precision']:.4f}",
                    f"{r['f1']:.4f}",
                ])
            pdf.metric_table(
                ["Target Recall", "Threshold", "Actual Recall", "Precision", "F1"],
                rows,
            )

    pdf.body_text(
        "Observations:"
    )
    pdf.bullet("XGBoost achieves recall targets with much lower thresholds (0.01-0.12), "
               "indicating better-calibrated probability estimates for the positive class.")
    pdf.bullet("Logistic Regression and Random Forest require very high thresholds "
               "(0.58-0.94), suggesting their predicted probabilities are heavily compressed "
               "toward the extremes.")
    pdf.bullet("LightGBM assigns all fraud predictions at threshold 1.0, meaning its "
               "probability estimates are poorly calibrated for this dataset.")
    pdf.bullet("At the 50% recall operating point, XGBoost flags 3,319 / 74,792 test "
               "transactions (4.44%) as potentially fraudulent.")

    # -- 9. TEST PREDICTIONS ---------------------------------------------
    pdf.add_page()
    pdf.chapter_title("9. Test Predictions")
    pdf.body_text(
        "Final test predictions are generated using the best model (XGBoost) at the 50% "
        "recall operating point. The threshold is calibrated on the validation set."
    )
    pdf.metric_table(
        ["Property", "Value"],
        [
            ["Best model", "XGBoost"],
            ["Decision threshold", "0.0247"],
            ["Target recall", "~50%"],
            ["Actual validation recall", "55.56%"],
            ["Validation precision", "1.40%"],
            ["Test transactions flagged", "3,319 / 74,792 (4.44%)"],
            ["Output file", "test_predictions.csv"],
        ],
    )
    pdf.body_text(
        "The test_predictions.csv file contains three columns: transactionid, "
        "fraud_probability (continuous score 0-1), and fraud_prediction (binary 0/1 "
        "using the chosen threshold)."
    )

    # -- 10. CONCLUSIONS -------------------------------------------------
    pdf.add_page()
    pdf.chapter_title("10. Conclusions & Recommendations")

    pdf.chapter_title("10.1 Summary of Findings", level=2)
    pdf.bullet("The pipeline correctly handles extreme class imbalance (0.11% fraud rate) "
               "through both algorithmic strategies and evaluation metric selection.")
    pdf.bullet("Temporal splitting prevents data leakage and provides realistic validation "
               "performance estimates.")
    pdf.bullet("Feature engineering adds meaningful fraud signals: velocity features, "
               "geographic mismatch flags, and domain frequency encoding.")
    pdf.bullet("XGBoost is the best performer on PR-AUC, the most appropriate metric for "
               "imbalanced fraud detection.")
    pdf.bullet("All models show relatively low precision at operational recall levels, "
               "which is expected given the extreme imbalance.")
    pdf.ln(3)

    pdf.chapter_title("10.2 Recommendations for Improvement", level=2)
    pdf.bullet("Hyperparameter tuning: Run Optuna optimisation (TUNE_HYPERPARAMS=1) to "
               "reduce XGBoost overfitting while maintaining or improving PR-AUC.")
    pdf.bullet("Early stopping: Add validation-set-based early stopping to gradient "
               "boosting models to prevent overfitting on 500 estimator rounds.")
    pdf.bullet("Cross-validation model selection: Enable RUN_CV_SELECTION=1 for more "
               "robust model selection using 3-fold TimeSeriesSplit.")
    pdf.bullet("Additional features: Card BIN analysis, transaction amount percentile "
               "within merchant, time since last transaction, cardholder age features.")
    pdf.bullet("Probability calibration: Apply Platt scaling or isotonic regression to "
               "improve probability estimate quality, especially for LightGBM.")
    pdf.bullet("Ensemble methods: Combine top models (XGBoost + Logistic Regression) "
               "via stacking or probability averaging for better generalisation.")
    pdf.bullet("Monitoring: Deploy with a drift detection module to flag concept drift "
               "in the transaction distribution over time.")
    pdf.ln(3)

    pdf.chapter_title("10.3 Pipeline Architecture", level=2)
    pdf.body_text(
        "The pipeline is modular and production-ready with clear separation of concerns:"
    )
    pdf.bullet("src/config.py -- Centralised configuration (features, paths, hyperparams)")
    pdf.bullet("src/data_quality.py -- 6-step data cleaning pipeline")
    pdf.bullet("src/features.py -- Leakage-safe feature engineering with lookup tables")
    pdf.bullet("src/model.py -- Model definitions, temporal splitting, optional tuning")
    pdf.bullet("src/evaluation.py -- Comprehensive evaluation with per-model plots")
    pdf.bullet("main.py -- Orchestrator with 10-step end-to-end workflow")
    pdf.ln(2)
    pdf.body_text(
        "Serialised artifacts (best_model.joblib, lookup_tables.joblib) enable "
        "deployment without re-training. The lookup tables ensure feature engineering "
        "at inference time uses only training-set statistics."
    )

    # -- Save ------------------------------------------------------------
    pdf.output(REPORT_PATH)
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    build_report()
