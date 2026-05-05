"""
Pairwise cross-era modeling.

The cross-era research goal — train a model on dataset A, evaluate it on
datasets B/C/D — runs into a feature-availability problem: every (train,
test) pair has a different "safe" feature set (twibot_2020 has no
in_reply_to_user_id, cresci_2017 has no follow graph, etc.). The FE step
emits a `pairwise_feature_schema.json` that resolves this per pair.

This module trains one full pipeline per (train, test) pair using exactly
that pair's feature columns. Each pickle is a self-contained
`sklearn.Pipeline` with imputer + scaler + classifier, plus a sidecar of
the feature columns and dataset metadata so a future inference run can
validate column alignment instead of silently reordering.

Conventions:
    - For a cross-era pair (train != test) we train on the full train_ds
      and evaluate on the full test_ds.
    - For a within-dataset pair (train == test, the diagonal of the
      experiment matrix) we use a stratified 80/20 split inside that
      dataset so the diagonal AUC reflects held-out performance, not a
      training-set fit.
    - Labels are mapped {bot=1, human=0}.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


WITHIN_DATASET_SPLIT_SIZE = 0.20
WITHIN_DATASET_SPLIT_SEED = 42


# ============================================================
# Data loading
# ============================================================

def load_features_and_schema(parquet_path, pairwise_schema_path):
    """
    Load the engineered feature parquet and the pairwise schema contract.

    Returns:
        (df, schema) where `schema[f"{train}__on__{test}"]` carries the
        column list to use for that pair.
    """
    df = pd.read_parquet(parquet_path)
    with open(pairwise_schema_path) as f:
        schema = json.load(f)
    return df, schema


def _coerce_label(label_series):
    """Map {bot, human} → {1, 0}. Anything else → NaN."""
    s = label_series.astype("string").str.lower().str.strip()
    out = pd.Series(np.nan, index=s.index, dtype="float")
    out[s == "bot"] = 1.0
    out[s == "human"] = 0.0
    return out


def get_pair_data(
    df, schema, train_ds, test_ds,
    label_col="label",
    split_size=WITHIN_DATASET_SPLIT_SIZE,
    split_seed=WITHIN_DATASET_SPLIT_SEED,
):
    """
    Return (X_train, y_train, X_test, y_test, feature_cols) for one pair.

    For train==test we do a stratified 80/20 split inside the dataset so the
    diagonal isn't a meaningless training-set AUC. For train!=test we use
    the full train and full test sets (no sub-sampling), since the whole
    point of cross-era evaluation is to use as much signal as possible.
    """
    key = f"{train_ds}__on__{test_ds}"
    if key not in schema:
        raise KeyError(
            f"No schema entry for {key!r}. "
            f"Re-run notebook 02 to regenerate pairwise_feature_schema.json."
        )

    feature_cols = list(schema[key]["kept"])

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Schema lists {len(missing)} feature columns not present in "
            f"the parquet (e.g. {missing[:5]}). "
            f"Schema and feature parquet are out of sync — re-run notebook 02."
        )

    if train_ds == test_ds:
        sub = df[df["dataset"] == train_ds].copy()
        y = _coerce_label(sub[label_col])
        X = sub[feature_cols]
        valid = y.notna()
        X, y = X[valid], y[valid].astype(int)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y,
            test_size=split_size,
            random_state=split_seed,
            stratify=y,
        )
        return X_tr, y_tr, X_te, y_te, feature_cols

    train_df = df[df["dataset"] == train_ds]
    test_df  = df[df["dataset"] == test_ds]

    y_tr = _coerce_label(train_df[label_col])
    y_te = _coerce_label(test_df[label_col])

    X_tr = train_df[feature_cols][y_tr.notna()]
    y_tr = y_tr.dropna().astype(int)
    X_te = test_df[feature_cols][y_te.notna()]
    y_te = y_te.dropna().astype(int)

    return X_tr, y_tr, X_te, y_te, feature_cols


# ============================================================
# Model factory
# ============================================================

def get_classifier(model_type="logreg"):
    if model_type == "logreg":
        return LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    if model_type == "ridge":
        return LogisticRegression(
            penalty="l2", C=1.0, max_iter=1000, class_weight="balanced", n_jobs=-1,
        )
    if model_type == "lasso":
        return LogisticRegression(
            penalty="l1", solver="liblinear", C=0.5,
            max_iter=1000, class_weight="balanced",
        )
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=200, max_depth=None, n_jobs=-1,
            class_weight="balanced_subsample", random_state=42,
        )
    if model_type == "xgb":
        return XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", n_jobs=-1, random_state=42,
        )
    if model_type == "svm":
        return LinearSVC(C=1.0, class_weight="balanced", max_iter=2000)
    raise ValueError(f"Unknown model_type: {model_type!r}")


def make_pipeline(model_type):
    """
    Build a full sklearn Pipeline: median-impute, optionally scale, classify.

    The imputer is fit on training data only (no test leakage), and lives
    inside the pipeline so saved models carry their own preprocessing.
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_type in ("logreg", "ridge", "lasso", "svm"):
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", get_classifier(model_type)))
    return Pipeline(steps)


# ============================================================
# Train + evaluate one pair
# ============================================================

def _predict_proba(pipeline, X):
    """Predict probabilities, falling back to scaled decision_function for SVM."""
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    scores = pipeline.decision_function(X)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)


def train_and_evaluate_pair(df, schema, train_ds, test_ds, model_type="logreg"):
    """
    Train one pipeline on the (train_ds, test_ds) pair-specific feature set
    and evaluate on the matching test rows.

    Returns:
        bundle (dict): self-contained record suitable for joblib.dump,
            includes the fitted pipeline, feature column list, and metadata.
        metrics (dict): accuracy, AUC, n_features.
    """
    X_tr, y_tr, X_te, y_te, feature_cols = get_pair_data(
        df, schema, train_ds, test_ds,
    )

    pipeline = make_pipeline(model_type)
    pipeline.fit(X_tr, y_tr)

    y_pred  = pipeline.predict(X_te)
    y_proba = _predict_proba(pipeline, X_te)

    metrics = {
        "accuracy":   float(accuracy_score(y_te, y_pred)),
        "auc":        float(roc_auc_score(y_te, y_proba)),
        "n_features": len(feature_cols),
        "n_train":    int(len(X_tr)),
        "n_test":     int(len(X_te)),
        "scope":      "within-dataset" if train_ds == test_ds else "cross-era",
    }

    bundle = {
        "pipeline":      pipeline,
        "feature_cols":  feature_cols,
        "train_dataset": train_ds,
        "test_dataset":  test_ds,
        "model_type":    model_type,
        "metrics":       metrics,
    }
    return bundle, metrics


# ============================================================
# Full pairwise experiment
# ============================================================

def run_pairwise_experiment(df, schema, datasets, model_type="logreg", verbose=True):
    """
    Train and evaluate every (train, test) pair for one model type.

    Returns:
        models: dict[train_ds][test_ds] -> bundle
        results: dict[train_ds][test_ds] -> metrics
    """
    models  = {tr: {} for tr in datasets}
    results = {tr: {} for tr in datasets}

    for train_ds in datasets:
        for test_ds in datasets:
            bundle, metrics = train_and_evaluate_pair(
                df, schema, train_ds, test_ds, model_type=model_type,
            )
            models[train_ds][test_ds]  = bundle
            results[train_ds][test_ds] = metrics

            if verbose:
                print(
                    f"  {train_ds:<12} → {test_ds:<12} "
                    f"({metrics['scope']:<14} {metrics['n_features']:>2} feat) "
                    f"AUC={metrics['auc']:.4f}  Acc={metrics['accuracy']:.4f}"
                )

    return models, results


# ============================================================
# Persistence
# ============================================================

def _model_filename(model_type, train_ds, test_ds):
    return f"{model_type}__train_{train_ds}__test_{test_ds}.pkl"


def save_pairwise_models(models, model_dir, model_type):
    """
    Save every (train, test) pair's bundle as its own pickle. Each pickle
    is self-describing (model + feature_cols + metadata) so a future
    inference script doesn't need a separate manifest to use it correctly.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for train_ds, by_test in models.items():
        for test_ds, bundle in by_test.items():
            path = model_dir / _model_filename(model_type, train_ds, test_ds)
            joblib.dump(bundle, path)
            n += 1
    print(f"Saved {n} pickled bundles to {model_dir}")


def load_pairwise_models(model_dir, datasets, model_type):
    model_dir = Path(model_dir)
    models = {tr: {} for tr in datasets}
    for train_ds in datasets:
        for test_ds in datasets:
            path = model_dir / _model_filename(model_type, train_ds, test_ds)
            if path.exists():
                models[train_ds][test_ds] = joblib.load(path)
    return models


def predict_with_bundle(bundle, df):
    """
    Score arbitrary rows from a feature dataframe using a saved bundle.

    Asserts the dataframe has every column the bundle was trained on; if
    you've added or removed features in FE since training, this raises
    rather than silently misaligning.
    """
    feature_cols = bundle["feature_cols"]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing {len(missing)} columns the bundle was "
            f"trained on (e.g. {missing[:5]}). Re-run FE or pick a different model."
        )
    X = df[feature_cols]  # exact column order from training
    return _predict_proba(bundle["pipeline"], X)


# ============================================================
# Result aggregation
# ============================================================

def results_to_matrix(all_results, datasets, metric="auc"):
    """Pivot the nested results dict into a (train, test) matrix."""
    return pd.DataFrame(
        {
            train: {
                test: all_results[train][test][metric]
                for test in datasets
            }
            for train in datasets
        }
    ).T  # rows = train_ds, cols = test_ds


def get_feature_importance(bundle):
    """
    Return a feature_importance Series for one bundle, sorted by |value|
    for linear models and by raw value for tree models.
    """
    clf = bundle["pipeline"].named_steps["clf"]
    cols = bundle["feature_cols"]
    model_type = bundle["model_type"]

    if model_type in ("logreg", "ridge", "lasso", "svm"):
        s = pd.Series(clf.coef_[0], index=cols)
        return s.reindex(s.abs().sort_values(ascending=False).index)
    if model_type in ("rf", "xgb"):
        return pd.Series(clf.feature_importances_, index=cols).sort_values(ascending=False)
    raise ValueError(f"Unsupported model_type: {model_type!r}")

# ============================================================
# Convenience accessors (read-only helpers)
# ============================================================

def get_bundle(models, train_ds, test_ds):
    """
    Safe accessor for a single (train, test) bundle.

    Raises a clear error if the bundle is missing.
    """
    try:
        return models[train_ds][test_ds]
    except KeyError:
        raise KeyError(f"Missing bundle for {train_ds} → {test_ds}")


def flatten_models(models):
    """
    Convert nested models dict into a flat list of bundles.

    Returns:
        list of bundle dicts
    """
    out = []
    for train_ds, by_test in models.items():
        for test_ds, bundle in by_test.items():
            out.append(bundle)
    return out


# ============================================================
# Metric aggregation helpers
# ============================================================

def mean_cross_era_auc(results, datasets):
    """
    Compute mean AUC over off-diagonal (cross-era) pairs only.
    """
    mat = results_to_matrix(results, datasets, metric="auc")

    values = []
    for tr in datasets:
        for te in datasets:
            if tr != te:
                values.append(mat.loc[tr, te])

    return float(np.mean(values))


def diagonal_vs_cross_auc(results, datasets):
    """
    Returns:
        list of dicts with:
            train_dataset
            diagonal_auc
            mean_cross_auc
    """
    mat = results_to_matrix(results, datasets, metric="auc")

    out = []
    for tr in datasets:
        diag = mat.loc[tr, tr]

        cross_vals = [
            mat.loc[tr, te]
            for te in datasets if te != tr
        ]

        out.append({
            "train_dataset": tr,
            "diagonal_auc": float(diag),
            "mean_cross_auc": float(np.mean(cross_vals)),
        })

    return out


def auc_degradation_matrix(results, datasets):
    """
    Compute (diagonal AUC - pair AUC) matrix.
    """
    mat = results_to_matrix(results, datasets, metric="auc")

    diag = np.diag(mat.values)
    baseline = np.tile(diag[:, None], (1, len(datasets)))

    return pd.DataFrame(
        baseline - mat.values,
        index=datasets,
        columns=datasets,
    )


# ============================================================
# Feature importance aggregation
# ============================================================

def collect_feature_importances(models):
    """
    Gather feature importance Series for all bundles.

    Returns:
        dict[(train__on__test)] -> Series
    """
    out = {}

    for tr, by_test in models.items():
        for te, bundle in by_test.items():
            key = f"{tr}__on__{te}"
            out[key] = get_feature_importance(bundle)

    return out


def feature_importance_matrix(models, top_k=None):
    """
    Build a feature × pair matrix of importances.

    Args:
        top_k: keep only top_k features per pair (optional)

    Returns:
        DataFrame: rows = features, cols = pairs
    """
    all_imps = collect_feature_importances(models)

    dfs = []
    for pair, s in all_imps.items():
        if top_k:
            s = s.head(top_k)
        dfs.append(s.rename(pair))

    df = pd.concat(dfs, axis=1).fillna(0.0)
    return df


def stable_top_features(models, k=10):
    """
    Identify features that frequently appear in top-k across pairs.

    Returns:
        Series: feature -> count
    """
    all_imps = collect_feature_importances(models)

    counts = {}

    for s in all_imps.values():
        top = s.head(k).index
        for f in top:
            counts[f] = counts.get(f, 0) + 1

    return pd.Series(counts).sort_values(ascending=False)