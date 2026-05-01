from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "logistic_regression"
REPORT_PATH = REPO_ROOT / "report" / "Research Report" / "logistic_regression_results.md"
SUMMARY_CSV = ARTIFACT_ROOT / "summary_metrics.csv"


def _load_metrics(dataset_dir: Path) -> dict:
    with (dataset_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_top_features(dataset_dir: Path, limit: int = 5) -> list[str]:
    frame = pd.read_csv(dataset_dir / "coefficients.csv")
    return frame["feature"].head(limit).tolist()


def _dataset_label(metrics: dict) -> str:
    args = metrics.get("training_args", {})
    max_rows = args.get("max_rows")
    if max_rows:
        return f"{metrics['dataset']} (sampled {metrics['row_count']:,} rows)"
    return f"{metrics['dataset']} ({metrics['row_count']:,} rows)"


def _build_summary_rows(dataset_dirs: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for dataset_dir in dataset_dirs:
        metrics = _load_metrics(dataset_dir)
        top_features = _load_top_features(dataset_dir)
        test = metrics["test_metrics"]
        rows.append(
            {
                "dataset": metrics["dataset"],
                "rows_used": metrics["row_count"],
                "feature_count": metrics["feature_count"],
                "accuracy": test["accuracy"],
                "precision": test["precision"],
                "recall": test["recall"],
                "f1": test["f1"],
                "roc_auc": test["roc_auc"],
                "top_features": ", ".join(top_features),
            }
        )
    return rows


def _write_markdown(dataset_dirs: list[Path]) -> None:
    lines: list[str] = []
    lines.append("# Logistic Regression Results")
    lines.append("")
    lines.append("This summary captures the logistic regression baseline experiments for historical Twitter/X bot detection.")
    lines.append("")
    lines.append("## Experimental setup")
    lines.append("")
    lines.append("- Model: custom logistic regression implemented with `numpy` and trained with mini-batch gradient descent")
    lines.append("- Features: 27 account/profile features derived from the processed user tables")
    lines.append("- Evaluation split: stratified train/validation/test split with 70/10/20 proportions")
    lines.append("- Class imbalance handling: balanced sample weights during training")
    lines.append("- Important note: `twibot_2022` was run on a 100,000-row stratified sample to keep experimentation practical")
    lines.append("")
    lines.append("## Test-set results")
    lines.append("")
    lines.append("| Dataset | Rows used | Accuracy | Precision | Recall | F1 | ROC-AUC |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")

    for dataset_dir in dataset_dirs:
        metrics = _load_metrics(dataset_dir)
        test = metrics["test_metrics"]
        lines.append(
            "| "
            f"{metrics['dataset']} | "
            f"{metrics['row_count']:,} | "
            f"{test['accuracy']:.4f} | "
            f"{test['precision']:.4f} | "
            f"{test['recall']:.4f} | "
            f"{test['f1']:.4f} | "
            f"{test['roc_auc']:.4f} |"
        )

    lines.append("")
    lines.append("## Observations")
    lines.append("")

    for dataset_dir in dataset_dirs:
        metrics = _load_metrics(dataset_dir)
        top_features = _load_top_features(dataset_dir)
        test = metrics["test_metrics"]
        lines.append(
            f"- `{_dataset_label(metrics)}`: accuracy was `{test['accuracy']:.4f}` and ROC-AUC was `{test['roc_auc']:.4f}`. "
            f"The most influential features were `{', '.join(top_features[:3])}`."
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The logistic regression baseline performs very strongly on the Cresci datasets, moderately on TwiBot-20, "
        "and noticeably worse on the sampled TwiBot-22 data. This suggests that simple linear decision boundaries "
        "capture older bot patterns well, but the separation between bots and humans becomes weaker in newer datasets."
    )
    lines.append("")
    lines.append(
        "These results give the project a clean baseline for comparing against Random Forest and for evaluating whether "
        "bot characteristics drift over time."
    )
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    dataset_dirs = sorted(
        [
            path
            for path in ARTIFACT_ROOT.iterdir()
            if path.is_dir() and (path / "metrics.json").exists()
        ]
    )
    if not dataset_dirs:
        raise SystemExit("No logistic regression artifacts found.")

    summary_rows = _build_summary_rows(dataset_dirs)
    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)
    _write_markdown(dataset_dirs)
    print(f"Summary CSV written to: {SUMMARY_CSV}")
    print(f"Markdown report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
