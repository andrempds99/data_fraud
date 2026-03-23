import logging

import numpy as np
from sklearn.base import clone
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
from sklearn.ensemble import RandomForestClassifier
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
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="aucpr",
            random_state=SEED,
            n_jobs=-1,
        )),
    ])


def build_lightgbm(preprocessor):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            is_unbalance=True,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )),
    ])


# ── Training helpers ───────────────────────────────────────────────────────────

def train_model(pipeline, X_train, y_train, name="Model"):
    """Fit a pipeline and return it."""
    logger.info("  Training %s …", name)
    pipeline.fit(X_train, y_train)
    logger.info("  %s trained.", name)
    return pipeline


def train_all_models(X_train, y_train):
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
        trained[name] = train_model(pipe, X_train, y_train, name)

    return trained


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
            "n_estimators":      trial.suggest_int("n_estimators", 100, 800),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
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
    logger.info("Optuna best PR-AUC: %.4f | params: %s", study.best_value, best)

    best.update({"is_unbalance": True, "random_state": SEED, "n_jobs": -1, "verbose": -1})
    final_pipe = Pipeline([
        ("preprocessor", build_preprocessor(num_feats, cat_feats)),
        ("classifier", lgb.LGBMClassifier(**best)),
    ])
    final_pipe.fit(X_train, y_train)
    logger.info("Optuna-tuned LightGBM trained.")
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