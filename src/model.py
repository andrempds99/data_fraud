import logging

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import xgboost as xgb
import lightgbm as lgb

from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, SEED

logger = logging.getLogger(__name__)


# ── Preprocessing ──────────────────────────────────────────────────────────────

def build_preprocessor(numeric_features=None, categorical_features=None):
    """Build a ColumnTransformer that imputes + scales numerics and one-hot encodes categoricals."""
    num_feats = numeric_features or NUMERIC_FEATURES
    cat_feats = categorical_features or CATEGORICAL_FEATURES

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_feats),
            ("cat", categorical_transformer, cat_feats),
        ],
        remainder="drop",
    )
    return preprocessor


# ── Train / Validation splits ────────────────────────────────────────────

def split_data(X, y, test_size=0.2):
    """Stratified random split (kept for backward compatibility)."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )
    logger.info("Stratified split: Train=%d  Val=%d", len(X_train), len(X_val))
    logger.info("Val fraud rate: %.4f%%", y_val.mean() * 100)
    return X_train, X_val, y_train, y_val


def split_data_temporal(X, y, test_size=0.2):
    """Time-based split: sort by `timestamp`, keep the last *test_size* fraction.

    Since test_data.csv is a temporal hold-out (Apr 2020 onwards), using the
    most-recent training rows as validation more closely mirrors production
    conditions and avoids future data leaking into the training fold.
    """
    sort_col = "timestamp" if "timestamp" in X.columns else "authtimestamp"
    sorted_index = X[sort_col].sort_values().index
    X_sorted = X.loc[sorted_index]
    y_sorted = y.loc[sorted_index]

    split_n = int(len(X_sorted) * (1 - test_size))
    X_train = X_sorted.iloc[:split_n]
    X_val   = X_sorted.iloc[split_n:]
    y_train = y_sorted.iloc[:split_n]
    y_val   = y_sorted.iloc[split_n:]

    logger.info("Temporal split  : Train=%d  Val=%d", len(X_train), len(X_val))
    logger.info("Val fraud rate  : %.4f%%", y_val.mean() * 100)
    return X_train, X_val, y_train, y_val


# ── Model builders ─────────────────────────────────────────────────────────────

def build_logistic_regression(preprocessor):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=SEED,
        )),
    ])


def build_random_forest(preprocessor):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            max_depth=15,
            min_samples_leaf=20,
            random_state=SEED,
            n_jobs=-1,
        )),
    ])


def build_xgboost(preprocessor, pos_weight=1.0):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="aucpr",
            random_state=SEED,
            n_jobs=-1,
        )),
    ])


def build_lightgbm(preprocessor):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", lgb.LGBMClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.05,
            is_unbalance=True,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            metric="auc",
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )),
    ])


# ── Training helpers ───────────────────────────────────────────────────────────

def _needs_early_stopping(pipeline):
    """Check if the pipeline's classifier supports early stopping."""
    clf = pipeline.named_steps["classifier"]
    return isinstance(clf, (xgb.XGBClassifier, lgb.LGBMClassifier))


def train_model(pipeline, X_train, y_train, name="Model", X_val=None, y_val=None):
    """Fit a pipeline, using early stopping for boosting models when val data is provided."""
    logger.info("  Training %s …", name)

    if X_val is not None and y_val is not None and _needs_early_stopping(pipeline):
        # Pre-fit the preprocessor on training data, then transform both sets
        preprocessor = pipeline.named_steps["preprocessor"]
        X_train_t = preprocessor.fit_transform(X_train)
        X_val_t = preprocessor.transform(X_val)

        clf = pipeline.named_steps["classifier"]
        fit_params = {"eval_set": [(X_val_t, y_val)]}

        if isinstance(clf, lgb.LGBMClassifier):
            fit_params["callbacks"] = [
                lgb.early_stopping(stopping_rounds=100, first_metric_only=True, verbose=False),
                lgb.log_evaluation(period=0),
            ]
        elif isinstance(clf, xgb.XGBClassifier):
            # Set early stopping dynamically (not in constructor) so that
            # StackingClassifier can refit without an eval_set.
            clf.set_params(early_stopping_rounds=50)

        clf.fit(X_train_t, y_train, **fit_params)
        best_iter = getattr(clf, "best_iteration_", getattr(clf, "best_iteration", None))

        # Safety net: if boosting stopped extremely early the model is
        # essentially untrained.  Re-train with a fixed, modest budget.
        MIN_ITERS = 50
        if isinstance(clf, lgb.LGBMClassifier) and best_iter is not None and best_iter < MIN_ITERS:
            logger.warning("  %s early-stopped at iteration %d (< %d). "
                           "Re-training with n_estimators=%d (no early stopping).",
                           name, best_iter, MIN_ITERS, MIN_ITERS * 6)
            clf.set_params(n_estimators=MIN_ITERS * 6)
            clf.fit(X_train_t, y_train)  # no eval_set / callbacks
            logger.info("  %s re-trained with %d fixed iterations.", name, MIN_ITERS * 6)
        else:
            logger.info("  %s trained with early stopping (best iteration: %s).",
                         name, best_iter)
    else:
        pipeline.fit(X_train, y_train)
        logger.info("  %s trained.", name)
    return pipeline


def train_all_models(X_train, y_train, X_val=None, y_val=None):
    """Build and train all models; the preprocessor spec is cloned per pipeline."""
    num_feats = [c for c in NUMERIC_FEATURES if c in X_train.columns]
    cat_feats  = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

    # Build the preprocessor configuration ONCE, then clone it into each
    # pipeline so every model gets an independent (unfitted) copy.
    base_preprocessor = build_preprocessor(num_feats, cat_feats)
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model_defs = {
        "LogisticRegression": build_logistic_regression(clone(base_preprocessor)),
        "RandomForest":       build_random_forest(clone(base_preprocessor)),
        "XGBoost":            build_xgboost(clone(base_preprocessor), pos_weight),
        "LightGBM":           build_lightgbm(clone(base_preprocessor)),
    }

    trained = {}
    for name, pipe in model_defs.items():
        trained[name] = train_model(pipe, X_train, y_train, name, X_val, y_val)

    return trained


def build_stacking_ensemble(models, X_train, y_train, X_val=None, y_val=None):
    """Build a stacking ensemble from pre-trained base models.

    Uses a LogisticRegression meta-learner on the base models' predicted
    probabilities. ``cv='prefit'`` avoids refitting the base estimators
    (they are already trained with early stopping etc.).
    """
    estimators = [(name, model) for name, model in models.items()]

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEED
        ),
        cv="prefit",
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=1,
    )

    logger.info("  Training Stacking Ensemble …")
    stack.fit(X_train, y_train)
    logger.info("  Stacking Ensemble trained.")
    return stack


def calibrate_model(model, X_val, y_val, method="isotonic"):
    """Wrap a fitted model with probability calibration.

    Uses ``FrozenEstimator`` to prevent refitting the base model
    (scikit-learn ≥1.6 API).
    """
    logger.info("  Calibrating model probabilities (method=%s) …", method)
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(model), cv=3, method=method
    )
    calibrated.fit(X_val, y_val)
    logger.info("  Calibration complete.")
    return calibrated


# ── SMOTE resampling ──────────────────────────────────────────────────────────

def train_with_smote(pipeline, X_train, y_train, name="Model"):
    """Re-train a clone of *pipeline* using SMOTE oversampling.

    SMOTE synthesises minority-class examples via k-NN interpolation so the
    classifier trains on a balanced dataset.  Built-in class-weight /
    scale_pos_weight parameters are disabled to avoid double-compensating.
    """
    from imblearn.over_sampling import SMOTE

    logger.info("  Training %s with SMOTE …", name)
    preprocessor = clone(pipeline.named_steps["preprocessor"])
    X_t = preprocessor.fit_transform(X_train)

    smote = SMOTE(random_state=SEED)
    X_res, y_res = smote.fit_resample(X_t, y_train)
    logger.info("  SMOTE resampled: %d → %d rows  (fraud: %d → %d)",
                len(y_train), len(y_res),
                int(y_train.sum()), int(y_res.sum()))

    clf = clone(pipeline.named_steps["classifier"])
    # Disable built-in class rebalancing — SMOTE handles it externally.
    params = clf.get_params()
    if "scale_pos_weight" in params:
        clf.set_params(scale_pos_weight=1.0)
    if "is_unbalance" in params:
        clf.set_params(is_unbalance=False)
    if "class_weight" in params:
        clf.set_params(class_weight=None)
    # Remove early stopping (no eval_set provided for SMOTE training).
    if "early_stopping_rounds" in params and params["early_stopping_rounds"] is not None:
        clf.set_params(early_stopping_rounds=None)

    clf.fit(X_res, y_res)
    result = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    logger.info("  %s (SMOTE) trained.", name)
    return result


# ── Overfitting comparison baseline ───────────────────────────────────────────

def build_baseline_xgboost(preprocessor, pos_weight=1.0):
    """Build an unregularized XGBoost for overfitting demonstration.

    Compared to the production model this uses:
    - deeper trees (max_depth=12 vs 6)
    - no L1 / L2 regularisation
    - no subsampling / column sampling
    - fixed 500 iterations (no early stopping)
    """
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", xgb.XGBClassifier(
            n_estimators=500,
            max_depth=12,
            learning_rate=0.1,
            scale_pos_weight=pos_weight,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=0.0,
            eval_metric="aucpr",
            random_state=SEED,
            n_jobs=-1,
        )),
    ])


# ── Optuna hyperparameter tuning ──────────────────────────────────────────────

def tune_lightgbm(X_train, y_train, n_trials=50):
    """Use Optuna to search LightGBM hyperparameters; return a fitted Pipeline.

    The preprocessor is built inside the Optuna objective and evaluated via
    cross_val_score with a full Pipeline — so scaler stats are refit on each
    CV fold's training data and never see validation rows.

    Requires: ``pip install optuna>=3.0``
    Run via:  ``TUNE_HYPERPARAMS=1 python main.py``
    """
    try:
        import optuna
    except ImportError:
        raise ImportError("optuna is not installed. Run: pip install optuna>=3.0")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    num_feats = [c for c in NUMERIC_FEATURES if c in X_train.columns]
    cat_feats  = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 1200),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "is_unbalance": True,
            "random_state": SEED,
            "n_jobs": -1,
            "verbose": -1,
        }
        # Build a full Pipeline inside the objective so the preprocessor is
        # independently fit on each CV fold's training split.
        pipe = Pipeline([
            ("preprocessor", build_preprocessor(num_feats, cat_feats)),
            ("classifier", lgb.LGBMClassifier(**params)),
        ])
        scores = cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=1
        )
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    logger.info("Optuna LightGBM best PR-AUC: %.4f | params: %s", study.best_value, best)

    best.update({"is_unbalance": True, "random_state": SEED, "n_jobs": -1, "verbose": -1})
    final_pipe = Pipeline([
        ("preprocessor", build_preprocessor(num_feats, cat_feats)),
        ("classifier", lgb.LGBMClassifier(**best)),
    ])
    final_pipe.fit(X_train, y_train)
    logger.info("Optuna-tuned LightGBM trained.")
    return final_pipe


def tune_xgboost(X_train, y_train, n_trials=50):
    """Use Optuna to search XGBoost hyperparameters; return a fitted Pipeline.

    Requires: ``pip install optuna>=3.0``
    Run via:  ``TUNE_HYPERPARAMS=1 python main.py``
    """
    try:
        import optuna
    except ImportError:
        raise ImportError("optuna is not installed. Run: pip install optuna>=3.0")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    num_feats = [c for c in NUMERIC_FEATURES if c in X_train.columns]
    cat_feats  = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 1200),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 20),
            "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight":  pos_weight,
            "eval_metric": "aucpr",
            "random_state": SEED,
            "n_jobs": -1,
        }
        pipe = Pipeline([
            ("preprocessor", build_preprocessor(num_feats, cat_feats)),
            ("classifier", xgb.XGBClassifier(**params)),
        ])
        scores = cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=1
        )
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    logger.info("Optuna XGBoost best PR-AUC: %.4f | params: %s", study.best_value, best)

    best.update({
        "scale_pos_weight": pos_weight,
        "eval_metric": "aucpr",
        "random_state": SEED,
        "n_jobs": -1,
    })
    final_pipe = Pipeline([
        ("preprocessor", build_preprocessor(num_feats, cat_feats)),
        ("classifier", xgb.XGBClassifier(**best)),
    ])
    final_pipe.fit(X_train, y_train)
    logger.info("Optuna-tuned XGBoost trained.")
    return final_pipe


# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_validate_model(pipeline, X, y, cv_folds=5):
    """Run stratified k-fold CV and return scores."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    scores = cross_val_score(
        pipeline, X, y, cv=cv, scoring="average_precision", n_jobs=-1
    )
    logger.info("CV PR-AUC: %.4f ± %.4f", scores.mean(), scores.std())
    return scores

def cross_validate_temporal(pipeline, X, y, cv_folds=3):
    """Run time-series cross-validation (TimeSeriesSplit) and return scores.

    Assumes ``X`` is already sorted in temporal order (as produced by
    ``split_data_temporal``).  Uses ``n_jobs=1`` to avoid nested parallelism
    when called from inside a loop that already trains multiple models.
    """
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    scores = cross_val_score(
        pipeline, X, y, cv=tscv, scoring="average_precision", n_jobs=1
    )
    logger.info(
        "TimeSeriesCV(%d-fold) PR-AUC: %.4f ± %.4f", cv_folds, scores.mean(), scores.std()
    )
    return scores