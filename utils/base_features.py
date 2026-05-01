from pathlib import Path

import duckdb
import pandas as pd


ROUND1_TWEET_COLS = [
    "tweet_count_actual",
    "avg_text_length",
    "retweet_ratio",
    "avg_hashtags",
    "avg_mentions",
    "avg_urls",
]

ROUND1_FEATURE_COLS = [
    "followers_count",
    "friends_count",
    "statuses_count",
    "favourites_count",
    "listed_count",
    "account_age_days",
    "ff_ratio",
    "has_bio",
    "has_location",
    "has_default_pic",
    "tweet_count_actual",
    "tweets_per_day",
    "avg_text_length",
    "retweet_ratio",
    "avg_hashtags",
    "avg_mentions",
    "avg_urls",
]


def build_account_features(users_df, dataset_name, reference_date):
    df = users_df.copy()

    df["dataset"] = dataset_name
    df["user_id"] = df["user_id"].astype(str)

    # --- Numeric columns ---
    numeric_cols = [
        "followers_count",
        "friends_count",
        "statuses_count",
        "favourites_count",
        "listed_count",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Account age ---
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
        ref = pd.Timestamp(reference_date, tz="UTC")
        df["account_age_days"] = (ref - df["created_at"]).dt.days

    # --- Follower/friend ratio ---
    if {"followers_count", "friends_count"}.issubset(df.columns):
        df["ff_ratio"] = df["followers_count"] / (df["friends_count"] + 1)

    # --- Profile completeness ---
    if "description" in df.columns:
        df["has_bio"] = (
            df["description"].notna()
            & (df["description"].astype(str).str.strip() != "")
        )

    if "location" in df.columns:
        df["has_location"] = (
            df["location"].notna()
            & (df["location"].astype(str).str.strip() != "")
        )

    if "default_profile_image" in df.columns:
        df["has_default_pic"] = (
            df["default_profile_image"]
            .astype(str)
            .str.strip()
            .isin(["1", "True", "true"])
        )

    # --- Verified flag ---
    if "verified" in df.columns:
        df["verified"] = df["verified"].astype(bool)

    # --- DROP RAW COLUMNS ---
    drop_cols = [
        "screen_name",
        "description",
        "location",
        "created_at",
        "default_profile_image",
    ]

    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    return df

def build_tweet_aggregates(tweet_path):
    tweet_path = str(Path(tweet_path))

    con = duckdb.connect()

    query = f"""
        SELECT
            CAST(user_id AS VARCHAR) AS user_id,
            COUNT(*) AS tweet_count_actual,

            AVG(LENGTH(COALESCE(text, ''))) AS avg_text_length,

            AVG(
                CASE
                    WHEN STARTS_WITH(COALESCE(text, ''), 'RT @') THEN 1
                    ELSE 0
                END
            ) AS retweet_ratio,

            AVG(array_length(regexp_extract_all(COALESCE(text, ''), '#[A-Za-z0-9_]+'))) AS avg_hashtags,

            AVG(array_length(regexp_extract_all(COALESCE(text, ''), '@[A-Za-z0-9_]+'))) AS avg_mentions,

            AVG(array_length(regexp_extract_all(COALESCE(text, ''), 'http[^ ]+'))) AS avg_urls

        FROM read_csv_auto('{tweet_path}')
        GROUP BY CAST(user_id AS VARCHAR)
    """

    try:
        agg = con.execute(query).df()
    finally:
        con.close()

    return agg

def merge_round1_features(account_features, tweet_features):
    df = account_features.copy()
    tweet_features = tweet_features.copy()

    df["user_id"] = df["user_id"].astype(str)
    tweet_features["user_id"] = tweet_features["user_id"].astype(str)

    merged = df.merge(tweet_features, on="user_id", how="left")

    for col in ROUND1_TWEET_COLS:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    if {"tweet_count_actual", "account_age_days"}.issubset(merged.columns):
        merged["tweets_per_day"] = (
            merged["tweet_count_actual"] / (merged["account_age_days"] + 1)
        )

    return merged

def build_round1_features_for_dataset(dataset_name, paths):
    users = pd.read_csv(paths["users"], low_memory=False)
    users["user_id"] = users["user_id"].astype(str)

    account_features = build_account_features(
        users_df=users,
        dataset_name=dataset_name,
        reference_date=paths["reference_date"],
    )

    tweet_features = build_tweet_aggregates(paths["tweets"])

    round1 = merge_round1_features(account_features, tweet_features)

    return round1


def save_feature_frame(df, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def available_feature_cols(df, cols=ROUND1_FEATURE_COLS):
    return [c for c in cols if c in df.columns]


def print_round1_summary(df):
    print("Rows by dataset:")
    print(df["dataset"].value_counts())

    if "label" in df.columns:
        print("\nLabel distribution:")
        print(pd.crosstab(df["dataset"], df["label"], margins=True))

    feature_cols = available_feature_cols(df)

    print("\nRound 1 feature columns:")
    for col in feature_cols:
        print("-", col)

    print("\nMissing values in Round 1 feature columns:")
    print(df[feature_cols].isna().sum())