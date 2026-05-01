from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import warnings

import numpy as np
import pandas as pd


DATASET_FILES = {
    "cresci_2015": "cresci_2015/users_cresci_2015.csv",
    "cresci_2017": "cresci_2017/users_cresci_2017.csv",
    "twibot_2020": "twibot_2020/users_twibot_20.csv",
    "twibot_2022": "twibot_2022/users_twibot_22.csv",
}

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOTS = [
    REPO_ROOT / "data" / "processed",
    Path("/Users/hsla/Desktop/data/processed"),
]

NUMERIC_INPUTS = [
    "followers_count",
    "friends_count",
    "statuses_count",
    "favourites_count",
    "listed_count",
]


@dataclass
class PreparedDataset:
    features: pd.DataFrame
    labels: np.ndarray
    metadata: pd.DataFrame
    reference_date: str


def resolve_dataset_path(dataset: str, data_root: str | Path | None = None) -> Path:
    if dataset not in DATASET_FILES:
        raise ValueError(f"Unsupported dataset: {dataset}")

    roots: list[Path] = []
    if data_root is not None:
        roots.append(Path(data_root))
    roots.extend(DEFAULT_DATA_ROOTS)

    for root in roots:
        candidate = root / DATASET_FILES[dataset]
        if candidate.exists():
            return candidate

    searched = ", ".join(str(root) for root in roots)
    raise FileNotFoundError(
        f"Could not find dataset '{dataset}'. Searched: {searched}"
    )


def _clean_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.where(~cleaned.isin(["", "nan", "None"]), np.nan)
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n", ""}:
        return False
    return bool(value)


def _parse_created_at(series: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Could not infer format, so each element will be parsed individually",
            category=UserWarning,
        )
        return pd.to_datetime(series, errors="coerce", utc=True)


def infer_reference_date(users_df: pd.DataFrame) -> pd.Timestamp:
    created = _parse_created_at(users_df["created_at"])
    valid = created.dropna()
    if valid.empty:
        return pd.Timestamp("2026-05-01T00:00:00Z")
    latest = valid.max().ceil("D")
    return latest


def build_user_features(
    users_df: pd.DataFrame, reference_date: str | None = None
) -> PreparedDataset:
    df = users_df.copy()

    for column in NUMERIC_INPUTS:
        df[column] = _clean_numeric(df[column])

    created_at = _parse_created_at(df["created_at"])
    ref_ts = (
        pd.Timestamp(reference_date, tz="UTC")
        if reference_date is not None
        else infer_reference_date(df)
    )

    account_age_days = (ref_ts - created_at).dt.total_seconds() / 86400.0
    account_age_days = account_age_days.clip(lower=0)
    account_age_safe = account_age_days.fillna(account_age_days.median()).clip(lower=1.0)

    followers = df["followers_count"]
    friends = df["friends_count"]
    statuses = df["statuses_count"]
    favourites = df["favourites_count"]
    listed = df["listed_count"]

    description = df["description"].fillna("").astype(str)
    location = df["location"].fillna("").astype(str)
    screen_name = df["screen_name"].fillna("").astype(str)

    features = pd.DataFrame(index=df.index)

    features["followers_count"] = followers
    features["friends_count"] = friends
    features["statuses_count"] = statuses
    features["favourites_count"] = favourites
    features["listed_count"] = listed
    features["account_age_days"] = account_age_days
    features["ff_ratio"] = followers / (friends + 1.0)
    features["tweet_rate"] = statuses / account_age_safe
    features["favourites_rate"] = favourites / account_age_safe
    features["listed_per_1k_followers"] = 1000.0 * listed / (followers + 1.0)
    features["friend_follower_gap"] = followers - friends
    features["log_followers"] = np.log1p(followers.clip(lower=0))
    features["log_friends"] = np.log1p(friends.clip(lower=0))
    features["log_statuses"] = np.log1p(statuses.clip(lower=0))
    features["log_favourites"] = np.log1p(favourites.clip(lower=0))
    features["log_listed"] = np.log1p(listed.clip(lower=0))
    features["verified_flag"] = df["verified"].map(_parse_bool).astype(int)
    features["has_bio"] = description.str.strip().ne("").astype(int)
    features["has_location"] = location.str.strip().ne("").astype(int)
    features["has_default_pic"] = df["default_profile_image"].map(_parse_bool).astype(int)
    features["bio_length"] = description.str.len()
    features["bio_word_count"] = description.str.split().str.len()
    features["bio_has_url"] = description.str.contains(r"http|www\.", case=False, regex=True).astype(int)
    features["bio_digit_count"] = description.str.count(r"\d")
    features["location_length"] = location.str.len()
    features["screen_name_length"] = screen_name.str.len()
    features["screen_name_digit_ratio"] = (
        screen_name.str.count(r"\d") / screen_name.str.len().replace(0, np.nan)
    ).fillna(0.0)

    labels = df["label"].astype(str).str.strip().str.lower().map({"human": 0, "bot": 1})
    if labels.isna().any():
        bad_values = sorted(df.loc[labels.isna(), "label"].astype(str).unique())
        raise ValueError(f"Unexpected label values: {bad_values}")

    metadata = df[["user_id", "screen_name", "label", "subset", "source"]].copy()

    return PreparedDataset(
        features=features,
        labels=labels.to_numpy(dtype=np.int64),
        metadata=metadata,
        reference_date=ref_ts.isoformat(),
    )


def stratified_split_indices(
    labels: np.ndarray, test_size: float, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    rng = np.random.default_rng(random_state)
    train_indices: list[np.ndarray] = []
    test_indices: list[np.ndarray] = []

    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        n_test = max(1, int(round(len(indices) * test_size)))
        n_test = min(n_test, len(indices) - 1) if len(indices) > 1 else 1
        test_indices.append(indices[:n_test])
        train_indices.append(indices[n_test:])

    train = np.concatenate(train_indices)
    test = np.concatenate(test_indices)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def fit_imputer(train_df: pd.DataFrame) -> pd.Series:
    medians = train_df.median(numeric_only=True)
    return medians.fillna(0.0)


def apply_imputer(df: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    return df.fillna(medians)


def fit_standardizer(train_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    means = train_df.mean(numeric_only=True)
    stds = train_df.std(numeric_only=True).replace(0, 1.0).fillna(1.0)
    return means, stds


def apply_standardizer(
    df: pd.DataFrame, means: pd.Series, stds: pd.Series
) -> pd.DataFrame:
    return (df - means) / stds


def balanced_sample_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=2).astype(float)
    weights = np.zeros_like(labels, dtype=float)
    total = len(labels)
    for label, count in enumerate(counts):
        if count == 0:
            continue
        weights[labels == label] = total / (len(counts) * count)
    return weights


def prepare_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def maybe_sample_rows(
    users_df: pd.DataFrame, labels: np.ndarray, max_rows: int | None, random_state: int
) -> tuple[pd.DataFrame, np.ndarray]:
    if max_rows is None or len(users_df) <= max_rows:
        return users_df, labels

    rng = np.random.default_rng(random_state)
    sampled_parts: list[pd.DataFrame] = []
    sampled_labels: list[np.ndarray] = []

    for label in np.unique(labels):
        group = users_df[labels == label]
        n_group = max(1, int(round(max_rows * len(group) / len(users_df))))
        take = min(len(group), n_group)
        indices = rng.choice(group.index.to_numpy(), size=take, replace=False)
        sampled_group = group.loc[indices]
        sampled_parts.append(sampled_group)
        sampled_labels.append(np.full(len(sampled_group), label, dtype=np.int64))

    sampled_df = pd.concat(sampled_parts).sample(frac=1.0, random_state=random_state)
    sampled_label_array = sampled_df["label"].astype(str).str.strip().str.lower().map(
        {"human": 0, "bot": 1}
    ).to_numpy(dtype=np.int64)
    return sampled_df.reset_index(drop=True), sampled_label_array


def feature_rank_frame(feature_names: Iterable[str], values: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "feature": list(feature_names),
            "coefficient": values,
            "abs_coefficient": np.abs(values),
        }
    )
    return frame.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
