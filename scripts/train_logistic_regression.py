from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.logistic_regression import NumpyLogisticRegression, classification_metrics
from utils.modeling import (
    apply_imputer,
    apply_standardizer,
    balanced_sample_weights,
    build_user_features,
    feature_rank_frame,
    fit_imputer,
    fit_standardizer,
    maybe_sample_rows,
    prepare_output_dir,
    resolve_dataset_path,
    stratified_split_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a logistic regression bot detector with pure numpy/pandas."
    )
    parser.add_argument(
        "--dataset",
        default="cresci_2017",
        choices=["cresci_2015", "cresci_2017", "twibot_2020", "twibot_2022"],
        help="Processed dataset to use for training.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Optional root directory containing processed dataset folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/logistic_regression",
        help="Directory for model artifacts and evaluation outputs.",
    )
    parser.add_argument(
        "--reference-date",
        default=None,
        help="Optional UTC reference date for account-age features.",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--l2-strength", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional stratified sample cap for faster experiments.",
    )
    return parser.parse_args()


def _save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    dataset_path = resolve_dataset_path(args.dataset, args.data_root)
    output_dir = prepare_output_dir(Path(args.output_dir) / args.dataset)

    print(f"Loading dataset: {args.dataset}")
    print(f"File: {dataset_path}")
    users_df = pd.read_csv(dataset_path)
    print(f"Rows loaded: {len(users_df):,}")

    labels = (
        users_df["label"].astype(str).str.strip().str.lower().map({"human": 0, "bot": 1})
    ).to_numpy(dtype=np.int64)
    users_df, labels = maybe_sample_rows(
        users_df, labels, max_rows=args.max_rows, random_state=args.random_state
    )
    if args.max_rows is not None:
        print(f"Rows after sampling: {len(users_df):,}")

    prepared = build_user_features(users_df, reference_date=args.reference_date)
    train_val_idx, test_idx = stratified_split_indices(
        prepared.labels, test_size=args.test_size, random_state=args.random_state
    )
    relative_val_size = args.val_size / (1.0 - args.test_size)
    train_idx, val_idx = stratified_split_indices(
        prepared.labels[train_val_idx],
        test_size=relative_val_size,
        random_state=args.random_state + 1,
    )
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]

    X_train_df = prepared.features.iloc[train_idx].reset_index(drop=True)
    X_val_df = prepared.features.iloc[val_idx].reset_index(drop=True)
    X_test_df = prepared.features.iloc[test_idx].reset_index(drop=True)
    y_train = prepared.labels[train_idx]
    y_val = prepared.labels[val_idx]
    y_test = prepared.labels[test_idx]

    medians = fit_imputer(X_train_df)
    X_train_df = apply_imputer(X_train_df, medians)
    X_val_df = apply_imputer(X_val_df, medians)
    X_test_df = apply_imputer(X_test_df, medians)

    means, stds = fit_standardizer(X_train_df)
    X_train = apply_standardizer(X_train_df, means, stds).to_numpy(dtype=float)
    X_val = apply_standardizer(X_val_df, means, stds).to_numpy(dtype=float)
    X_test = apply_standardizer(X_test_df, means, stds).to_numpy(dtype=float)

    train_weights = balanced_sample_weights(y_train)
    val_weights = balanced_sample_weights(y_val)

    model = NumpyLogisticRegression(
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        l2_strength=args.l2_strength,
        patience=args.patience,
        random_state=args.random_state,
    )
    model.fit(
        X_train,
        y_train,
        sample_weight=train_weights,
        X_val=X_val,
        y_val=y_val,
        val_weight=val_weights,
    )

    val_prob = model.predict_proba(X_val)
    test_prob = model.predict_proba(X_test)
    val_metrics = classification_metrics(y_val, val_prob, threshold=args.threshold)
    test_metrics = classification_metrics(y_test, test_prob, threshold=args.threshold)

    print("Validation metrics:")
    for key, value in val_metrics.items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")

    print("Test metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")

    coefficient_frame = feature_rank_frame(prepared.features.columns, model.coef_)
    coefficient_frame.to_csv(output_dir / "coefficients.csv", index=False)

    history_frame = pd.DataFrame(
        [
            {
                "epoch": row.epoch,
                "train_loss": row.train_loss,
                "val_loss": row.val_loss,
            }
            for row in model.history_
        ]
    )
    history_frame.to_csv(output_dir / "training_history.csv", index=False)

    test_predictions = prepared.metadata.iloc[test_idx].reset_index(drop=True).copy()
    test_predictions["bot_probability"] = test_prob
    test_predictions["predicted_label"] = np.where(
        test_prob >= args.threshold, "bot", "human"
    )
    test_predictions["actual_label"] = np.where(y_test == 1, "bot", "human")
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    model_payload = {
        "dataset": args.dataset,
        "dataset_path": str(dataset_path),
        "reference_date": prepared.reference_date,
        "feature_names": list(prepared.features.columns),
        "medians": medians.to_dict(),
        "means": means.to_dict(),
        "stds": stds.to_dict(),
        "intercept": model.intercept_,
        "coefficients": model.coef_,
        "threshold": args.threshold,
        "training_args": vars(args),
    }
    with (output_dir / "model.pkl").open("wb") as handle:
        pickle.dump(model_payload, handle)

    summary = {
        "dataset": args.dataset,
        "dataset_path": str(dataset_path),
        "reference_date": prepared.reference_date,
        "row_count": int(len(users_df)),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "feature_count": int(prepared.features.shape[1]),
        "training_args": vars(args),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    _save_json(output_dir / "metrics.json", summary)

    top_features = coefficient_frame.head(10)
    print("Top coefficients:")
    print(top_features.to_string(index=False))
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
