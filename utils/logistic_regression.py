from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40, 40)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass
class TrainingHistory:
    epoch: int
    train_loss: float
    val_loss: float | None


class NumpyLogisticRegression:
    def __init__(
        self,
        learning_rate: float = 0.05,
        epochs: int = 400,
        batch_size: int = 512,
        l2_strength: float = 0.001,
        patience: int = 25,
        tolerance: float = 1e-5,
        random_state: int = 42,
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2_strength = l2_strength
        self.patience = patience
        self.tolerance = tolerance
        self.random_state = random_state
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.history_: list[TrainingHistory] = []

    def _loss(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        probs = self.predict_proba(X)
        probs = np.clip(probs, 1e-9, 1.0 - 1e-9)
        losses = -(y * np.log(probs) + (1 - y) * np.log(1 - probs))
        if sample_weight is not None:
            losses = losses * sample_weight
            denom = sample_weight.sum()
        else:
            denom = len(y)
        reg = 0.5 * self.l2_strength * float(np.dot(self.coef_, self.coef_))
        return float(losses.sum() / max(denom, 1.0) + reg)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        val_weight: np.ndarray | None = None,
    ) -> "NumpyLogisticRegression":
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)
        self.coef_ = np.zeros(n_features, dtype=float)
        self.intercept_ = 0.0

        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=float)

        best_loss = np.inf
        best_coef = self.coef_.copy()
        best_intercept = self.intercept_
        patience_left = self.patience

        for epoch in range(1, self.epochs + 1):
            indices = rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                batch = indices[start : start + self.batch_size]
                xb = X[batch]
                yb = y[batch]
                wb = sample_weight[batch]

                logits = xb @ self.coef_ + self.intercept_
                probs = _sigmoid(logits)
                errors = probs - yb

                denom = max(wb.sum(), 1.0)
                weighted_errors = errors * wb
                grad_w = (xb.T @ weighted_errors) / denom + self.l2_strength * self.coef_
                grad_b = float(weighted_errors.sum() / denom)

                self.coef_ -= self.learning_rate * grad_w
                self.intercept_ -= self.learning_rate * grad_b

            train_loss = self._loss(X, y, sample_weight)
            val_loss = None
            tracked_loss = train_loss
            if X_val is not None and y_val is not None:
                val_loss = self._loss(X_val, y_val, val_weight)
                tracked_loss = val_loss

            self.history_.append(
                TrainingHistory(epoch=epoch, train_loss=train_loss, val_loss=val_loss)
            )

            if tracked_loss + self.tolerance < best_loss:
                best_loss = tracked_loss
                best_coef = self.coef_.copy()
                best_intercept = self.intercept_
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        self.coef_ = best_coef
        self.intercept_ = best_intercept
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model has not been fit yet.")
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(self.decision_function(X))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int64)


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def roc_auc_score_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")

    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    positive_rank_sum = ranks[y_true == 1].sum()
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    accuracy = _safe_divide(tp + tn, len(y_true))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score_manual(y_true, y_prob),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
    }
