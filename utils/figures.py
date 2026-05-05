"""
Centralized figure output for the project.

Every figure produced by any notebook should go through `save_fig()` (or
`savefig()` as an alias). This guarantees:
    - all figures land in `report/figures/` regardless of which notebook
      created them or what the working directory is
    - consistent dpi / bbox / format across the report
    - the report's LaTeX `\\includegraphics` paths stay stable

Usage:
    from utils.figures import save_fig

    fig, ax = plt.subplots()
    ax.plot(...)
    save_fig("01_eda__label_balance")     # -> report/figures/01_eda__label_balance.png

Naming convention: `{notebook_prefix}__{descriptor}` so figures sort by
notebook in the directory and are easy to grep from the LaTeX source.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Optional: only imported when needed (avoid hard dependency at import time)
from utils.modeling import predict_with_bundle, get_feature_importance

# ------------------------------------------------------------
# Global conventions
# ------------------------------------------------------------

DATASETS = [
    "cresci_2015",
    "cresci_2017",
    "twibot_2020",
    "twibot_2022",
]

DATASET_LABELS = {
    "cresci_2015": "Cresci 2015",
    "cresci_2017": "Cresci 2017",
    "twibot_2020": "Twibot 2020",
    "twibot_2022": "Twibot 2022",
}

LABEL_COLORS = {
    "bot": "#d62728",
    "human": "#1f77b4",
}

CMAP_SEQ = "viridis"
CMAP_DIVERGING = "RdYlGn"


# Resolved once at import: walk up from this file until we hit a directory
# that contains both `report/` and `data/`. That's the project root,
# regardless of which notebook (or subdir of notebooks) we're called from.
def _find_project_root():
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        if (parent / "report").is_dir() and (parent / "data").is_dir():
            return parent
    # Fallback: assume utils/ is one level under the project root.
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = _find_project_root()
FIGURES_DIR  = PROJECT_ROOT / "report" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Defaults tuned for the report. Override per-call if needed.
DEFAULT_FORMATS = ("png",)
DEFAULT_DPI     = 200


def save_fig(
    name,
    fig=None,
    formats=DEFAULT_FORMATS,
    dpi=DEFAULT_DPI,
    tight=True,
    show=True,
    transparent=False,
):
    """
    Save the current matplotlib figure (or a passed `fig`) into
    `report/figures/{name}.{format}` for every requested format.

    Args:
        name: filename stem, no extension. Slashes are flattened to `__`.
        fig: matplotlib Figure. Defaults to the current figure.
        formats: iterable of file extensions to write (e.g. ("png", "pdf")).
        dpi: rasterization dpi for png.
        tight: call `tight_layout()` before saving.
        show: call `plt.show()` after saving (notebooks expect this).
        transparent: save with transparent background.

    Returns:
        list of `Path`s written.
    """
    if fig is None:
        fig = plt.gcf()

    safe_name = name.replace("/", "__").replace("\\", "__")

    if tight:
        fig.tight_layout()

    written = []
    for ext in formats:
        out_path = FIGURES_DIR / f"{safe_name}.{ext}"
        fig.savefig(
            out_path,
            dpi=dpi,
            bbox_inches="tight",
            transparent=transparent,
        )
        written.append(out_path)
        print(f"  saved figure: {out_path.relative_to(PROJECT_ROOT)}")

    if show:
        plt.show()

    return written


# Convenience alias matching matplotlib's naming.
savefig = save_fig


def list_figures():
    """Return a sorted list of figure paths currently in report/figures/."""
    return sorted(p for p in FIGURES_DIR.iterdir() if p.is_file())

# ------------------------------------------------------------
# Generic plotting primitives
# ------------------------------------------------------------

def plot_heatmap(
    matrix,
    x_labels,
    y_labels,
    title=None,
    cmap=CMAP_DIVERGING,
    annot=True,
    fmt=".2f",
    vmin=None,
    vmax=None,
):
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticklabels(y_labels)

    if title:
        ax.set_title(title)

    if annot:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                ax.text(j, i, format(val, fmt),
                        ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    return fig, ax


def plot_lines(x_labels, series_dict, title=None, ylabel=None):
    fig, ax = plt.subplots(figsize=(6, 4))

    for name, y in series_dict.items():
        ax.plot(x_labels, y, marker="o", label=name)

    ax.legend()
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45)

    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)

    return fig, ax


def plot_bar(values, labels, title=None, ylabel=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")

    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)

    return fig, ax


def plot_histogram(data_dict, bins=50, log=False, title=None):
    fig, ax = plt.subplots(figsize=(6, 4))

    for label, values in data_dict.items():
        ax.hist(values, bins=bins, alpha=0.5, label=label)

    if log:
        ax.set_xscale("log")

    ax.legend()

    if title:
        ax.set_title(title)

    return fig, ax


# ------------------------------------------------------------
# Matrix + story builders (AUC, degradation, etc.)
# ------------------------------------------------------------

def build_auc_heatmap(df_matrix, title):
    mat = df_matrix.loc[DATASETS, DATASETS].values

    fig, _ = plot_heatmap(
        mat,
        x_labels=[DATASET_LABELS[d] for d in DATASETS],
        y_labels=[DATASET_LABELS[d] for d in DATASETS],
        title=title,
        vmin=0.5,
        vmax=1.0,
    )
    return fig


def build_bots_over_time(df_matrix, title):
    series = {}

    for train_ds in DATASETS:
        series[DATASET_LABELS[train_ds]] = [
            df_matrix.loc[train_ds, test_ds]
            for test_ds in DATASETS
        ]

    fig, _ = plot_lines(
        x_labels=[DATASET_LABELS[d] for d in DATASETS],
        series_dict=series,
        title=title,
        ylabel="AUC",
    )
    return fig


def build_auc_degradation(df_matrix):
    diag = np.diag(df_matrix.loc[DATASETS, DATASETS])
    baseline = np.tile(diag[:, None], (1, len(DATASETS)))

    degradation = baseline - df_matrix.loc[DATASETS, DATASETS].values

    fig, _ = plot_heatmap(
        degradation,
        x_labels=[DATASET_LABELS[d] for d in DATASETS],
        y_labels=[DATASET_LABELS[d] for d in DATASETS],
        title="AUC Degradation",
        cmap="coolwarm",
        vmin=0,
    )
    return fig


# ------------------------------------------------------------
# Model-based plots (uses bundles)
# ------------------------------------------------------------

def predict_labels_and_proba(bundle, df):
    from utils.modeling import predict_with_bundle

    proba = predict_with_bundle(bundle, df)
    preds = (proba >= 0.5).astype(int)
    return preds, proba


def build_confusion_matrix(bundle, df, normalize=False):
    from sklearn.metrics import confusion_matrix

    preds, _ = predict_labels_and_proba(bundle, df)

    y_true = df["label"].map({"bot": 1, "human": 0}).dropna().astype(int)

    cm = confusion_matrix(y_true, preds)

    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True)

    fig, _ = plot_heatmap(
        cm,
        x_labels=["Pred Human", "Pred Bot"],
        y_labels=["True Human", "True Bot"],
        title=f"{bundle['train_dataset']} → {bundle['test_dataset']}",
        cmap="Blues",
        annot=True,
        fmt=".2f" if normalize else "d",
    )

    return fig


def build_roc_curve(bundle, df):
    from sklearn.metrics import roc_curve, auc

    _, proba = predict_labels_and_proba(bundle, df)

    y_true = df["label"].map({"bot": 1, "human": 0}).dropna().astype(int)

    fpr, tpr, _ = roc_curve(y_true, proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{bundle['train_dataset']} → {bundle['test_dataset']}")
    ax.legend()

    return fig


def build_feature_importance(bundle, top_k=15):
    from utils.modeling import get_feature_importance

    s = get_feature_importance(bundle)
    s = s.head(top_k)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(s.index[::-1], s.values[::-1])

    ax.set_title(
        f"{bundle['model_type']} | "
        f"{bundle['train_dataset']} → {bundle['test_dataset']}"
    )

    return fig