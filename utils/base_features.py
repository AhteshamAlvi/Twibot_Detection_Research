"""
Round 1 base features: account metadata + per-user tweet aggregates.

Contract for the cross-dataset evaluation goal:

  - Every feature in `ROUND1_FEATURE_COLS` is materialized for every dataset.
  - Where the source data cannot support a feature (e.g. twibot_2020 has no
    in_reply_to_user_id, so reply_ratio cannot be computed), the column is
    set to NaN for that dataset rather than 0. This avoids the
    "missing-data masquerades as a behavioral signal" leakage.
  - User IDs are strings everywhere; tweet IDs are strings; reply IDs are
    strings. No CSV read is allowed to silently upcast a 19-digit Twitter ID
    to float and stringify as `4.5e+18`.
  - Heavy-tailed counts (followers, friends, statuses, listed, tweet_count)
    get explicit log1p companions for linear-model use.
  - tweets_per_day is computed from the user's actual tweet timespan when
    timestamps are present; otherwise we fall back to a clipped
    account_age_days denominator.

The per-dataset availability of features lives in `DATASET_FEATURE_AVAILABILITY`,
keyed by dataset name. Anything False there gets NaN'd post-merge.
"""

from pathlib import Path
import warnings

import duckdb
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Canonical feature contract
# ---------------------------------------------------------------------------

# Account metadata features — derived from the users table.
ACCOUNT_FEATURE_COLS = [
    "followers_count",
    "friends_count",
    "statuses_count",
    "listed_count",
    "log1p_followers",
    "log1p_friends",
    "log1p_statuses",
    "log1p_listed",
    "followers_per_day",
    "friends_per_day",
    "statuses_per_day",
    "listed_per_day",
    "account_age_days",
    "ff_ratio",
    "has_bio",
    "has_location",
    "has_default_pic",
    "verified",
]

# Tweet behaviour features — derived from the tweets table.
TWEET_FEATURE_COLS = [
    "tweet_count_actual",
    "log1p_tweet_count",
    "tweets_per_day",
    "tweet_active_days",
    "avg_text_length",
    "retweet_ratio",
    "avg_hashtags",
    "avg_mentions",
    "avg_urls",
    "reply_ratio",
    "has_tweet_data",
]

# Full Round 1 feature column list. All four datasets must end up with
# exactly these columns after Round 1.
ROUND1_FEATURE_COLS = ACCOUNT_FEATURE_COLS + TWEET_FEATURE_COLS

# Identifier and metadata columns that travel alongside features.
ROUND1_META_COLS = ["dataset", "user_id", "label", "subset", "source"]

# Per-dataset feature reliability. Anything set to False gets overwritten with
# NaN at validation time so downstream code can treat it as missing rather
# than as a 0-valued behavioural signal.
#
# Notes:
#   - twibot_2020 tweets are stored as plain strings: no created_at,
#     no tweet_id, no in_reply_to_user_id. So reply_ratio,
#     tweet_active_days and tweets_per_day-from-tweet-window are all
#     unrecoverable.
#   - twibot_2022 is v2 API. v2 returns retweets in `referenced_tweets`
#     (not as "RT @..." text), so a text-prefix heuristic is unreliable.
#     We mark retweet_ratio as unavailable until cleaning is updated to
#     extract the canonical signal.
DATASET_FEATURE_AVAILABILITY = {
    "cresci_2015": {
        "retweet_ratio": True,
        "reply_ratio": True,
        "tweet_active_days": True,
    },
    "cresci_2017": {
        "retweet_ratio": True,
        "reply_ratio": True,
        "tweet_active_days": True,
    },
    "twibot_2020": {
        "retweet_ratio": True,   # text-prefix heuristic still applies
        "reply_ratio": False,    # in_reply_to_user_id not preserved
        "tweet_active_days": False,
    },
    "twibot_2022": {
        "retweet_ratio": False,  # v2 API: RT @ heuristic unreliable
        "reply_ratio": True,
        "tweet_active_days": True,
    },
}


# ---------------------------------------------------------------------------
# IO helpers — string ID enforcement
# ---------------------------------------------------------------------------

# Always read these as strings to avoid float-upcast → "4.5e+18" stringification.
_STRING_ID_COLS = ["user_id", "tweet_id", "in_reply_to_user_id", "source_id", "target_id"]


def read_users_csv(path):
    """Read a pre-processed users CSV with user_id forced to string."""
    return pd.read_csv(
        path,
        low_memory=False,
        dtype={c: "string" for c in _STRING_ID_COLS},
    )


def _canonicalize_id_series(s):
    """
    Convert a series of mixed-type IDs to clean string IDs.

    Handles the three failure modes:
      - float upcast from NaN-containing int columns ("4.5e+18")
      - trailing ".0" from int-as-float ("123.0" → "123")
      - empty string / "nan" / "None"
    """
    if s.dtype.name == "string":
        out = s
    else:
        out = s.astype("string")

    out = out.where(~out.isin(["", "nan", "None", "NaN"]), pd.NA)

    # Strip ".0" suffix from things like "123.0"
    mask = out.notna() & out.str.match(r"^-?\d+\.0+$")
    out = out.mask(mask, out.str.replace(r"\.0+$", "", regex=True))

    return out


# ---------------------------------------------------------------------------
# Account features
# ---------------------------------------------------------------------------

def build_account_features(users_df, dataset_name, reference_date):
    """
    Build per-user account-metadata features.

    The output schema is exactly `ACCOUNT_FEATURE_COLS` plus
    `["dataset", "user_id", "label", "subset", "source"]` where present.
    Missing source columns are filled with NaN so the schema is uniform
    across datasets.
    """
    df = users_df.copy()

    df["dataset"] = dataset_name
    df["user_id"] = _canonicalize_id_series(df["user_id"])

    # --- Numeric counts ---
    for col in ["followers_count", "friends_count", "statuses_count", "listed_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    # --- log1p companions for heavy-tailed counts ---
    df["log1p_followers"] = np.log1p(df["followers_count"].clip(lower=0))
    df["log1p_friends"]   = np.log1p(df["friends_count"].clip(lower=0))
    df["log1p_statuses"]  = np.log1p(df["statuses_count"].clip(lower=0))
    df["log1p_listed"]    = np.log1p(df["listed_count"].clip(lower=0))

    # --- Account age (clipped at 0; NaN if created_at missing) ---
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
        ref = pd.Timestamp(reference_date, tz="UTC")
        age = (ref - df["created_at"]).dt.days
        df["account_age_days"] = age.clip(lower=0)
    else:
        df["account_age_days"] = np.nan

    # --- Follower/friend ratio (safe under NaN) ---
    df["ff_ratio"] = df["followers_count"] / (df["friends_count"].fillna(0) + 1)

    # --- Per-day rates (reference-date stable) ---
    # The raw counts and `account_age_days` both shift across eras: a 2022
    # human's follower count and account age are systematically larger than
    # a 2015 human's. The ratio counts/(age + 1) absorbs the shift, so it
    # transfers across datasets in a way the raw counts don't.
    age_denom = df["account_age_days"].clip(lower=0).fillna(0) + 1
    df["followers_per_day"] = df["followers_count"] / age_denom
    df["friends_per_day"]   = df["friends_count"]   / age_denom
    df["statuses_per_day"]  = df["statuses_count"]  / age_denom
    df["listed_per_day"]    = df["listed_count"]    / age_denom

    # --- Profile completeness ---
    def _truthy_with_na(s):
        """
        Coerce strings/bools to {1, 0, pd.NA} preserving NA semantics.
        Without this, NaN inputs collapse to 0 (a real signal — "user has
        no default pic"), which dataset-wide can become a zero-variance
        feature that the model learns as a dataset-identity proxy.
        """
        s = s.astype("string").str.strip().str.lower()
        out = pd.Series(pd.NA, index=s.index, dtype="Int8")
        out[s.isin(["1", "true"])] = 1
        out[s.isin(["0", "false"])] = 0
        return out

    if "description" in df.columns:
        df["has_bio"] = (
            df["description"].notna()
            & (df["description"].astype(str).str.strip() != "")
        ).astype("Int8")
    else:
        df["has_bio"] = pd.Series(pd.NA, index=df.index, dtype="Int8")

    if "location" in df.columns:
        df["has_location"] = (
            df["location"].notna()
            & (df["location"].astype(str).str.strip() != "")
        ).astype("Int8")
    else:
        df["has_location"] = pd.Series(pd.NA, index=df.index, dtype="Int8")

    if "default_profile_image" in df.columns:
        df["has_default_pic"] = _truthy_with_na(df["default_profile_image"])
    else:
        df["has_default_pic"] = pd.Series(pd.NA, index=df.index, dtype="Int8")

    if "verified" in df.columns:
        df["verified"] = _truthy_with_na(df["verified"])
    else:
        df["verified"] = pd.Series(pd.NA, index=df.index, dtype="Int8")

    # --- Drop raw columns no longer needed ---
    drop_cols = [
        "screen_name",
        "description",
        "location",
        "created_at",
        "default_profile_image",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


# ---------------------------------------------------------------------------
# Tweet aggregates
# ---------------------------------------------------------------------------

def _tweet_columns(tweet_path):
    """Peek at the CSV header to see which optional columns exist."""
    return list(pd.read_csv(tweet_path, nrows=0).columns)


def build_tweet_aggregates(tweet_path):
    """
    Compute per-user tweet aggregates from a pre-processed tweets CSV.

    Returns a dataframe with one row per user that appears in the tweet file
    and the columns:
        user_id, tweet_count_actual, avg_text_length,
        retweet_ratio, reply_ratio,
        avg_hashtags, avg_mentions, avg_urls,
        first_tweet_at, last_tweet_at, tweet_active_days

    Optional columns (created_at, in_reply_to_user_id, is_retweet, is_reply)
    are detected from the file header and used when present. Missing optional
    columns produce NaN-filled aggregates rather than misleading zeros.
    """
    tweet_path = str(Path(tweet_path))
    cols = _tweet_columns(tweet_path)

    has_created_at  = "created_at" in cols
    has_reply_id    = "in_reply_to_user_id" in cols
    has_is_retweet  = "is_retweet" in cols
    has_is_reply    = "is_reply" in cols

    # Build the column list lazily based on what the CSV actually has.
    # We always read user_id and text. Other columns added conditionally so
    # DuckDB doesn't complain about missing columns.
    select_cols = ["user_id", "text"]
    if has_created_at:
        select_cols.append("created_at")
    if has_reply_id:
        select_cols.append("in_reply_to_user_id")
    if has_is_retweet:
        select_cols.append("is_retweet")
    if has_is_reply:
        select_cols.append("is_reply")

    # Build the retweet/reply expressions based on what's available.
    if has_is_retweet:
        retweet_expr = "AVG(CAST(is_retweet AS INTEGER))"
    else:
        # Fallback heuristic — works for v1 datasets, unreliable for v2.
        retweet_expr = (
            "AVG(CASE WHEN STARTS_WITH(COALESCE(text, ''), 'RT @') "
            "THEN 1 ELSE 0 END)"
        )

    if has_is_reply:
        reply_expr = "AVG(CAST(is_reply AS INTEGER))"
    elif has_reply_id:
        reply_expr = (
            "AVG(CASE WHEN TRIM(COALESCE(CAST(in_reply_to_user_id AS VARCHAR), '')) "
            "NOT IN ('', '0', 'nan', 'None') THEN 1 ELSE 0 END)"
        )
    else:
        reply_expr = "CAST(NULL AS DOUBLE)"

    if has_created_at:
        timespan_select = """,
            MIN(TRY_CAST(created_at AS TIMESTAMP)) AS first_tweet_at,
            MAX(TRY_CAST(created_at AS TIMESTAMP)) AS last_tweet_at
        """
    else:
        timespan_select = """,
            CAST(NULL AS TIMESTAMP) AS first_tweet_at,
            CAST(NULL AS TIMESTAMP) AS last_tweet_at
        """

    query = f"""
        SELECT
            CAST(user_id AS VARCHAR) AS user_id,
            COUNT(*) AS tweet_count_actual,
            AVG(LENGTH(COALESCE(text, ''))) AS avg_text_length,
            {retweet_expr} AS retweet_ratio,
            {reply_expr} AS reply_ratio,
            AVG(array_length(regexp_extract_all(COALESCE(text, ''), '#[A-Za-z0-9_]+'))) AS avg_hashtags,
            AVG(array_length(regexp_extract_all(COALESCE(text, ''), '@[A-Za-z0-9_]+'))) AS avg_mentions,
            AVG(array_length(regexp_extract_all(COALESCE(text, ''), 'http[^ ]+'))) AS avg_urls
            {timespan_select}
        FROM read_csv_auto('{tweet_path}', SAMPLE_SIZE=-1)
        GROUP BY CAST(user_id AS VARCHAR)
    """

    con = duckdb.connect()
    try:
        agg = con.execute(query).df()
    finally:
        con.close()

    if has_created_at:
        agg["first_tweet_at"] = pd.to_datetime(agg["first_tweet_at"], errors="coerce", utc=True)
        agg["last_tweet_at"]  = pd.to_datetime(agg["last_tweet_at"],  errors="coerce", utc=True)
        span = (agg["last_tweet_at"] - agg["first_tweet_at"]).dt.total_seconds() / 86400
        agg["tweet_active_days"] = span.clip(lower=1)
    else:
        agg["tweet_active_days"] = np.nan

    return agg


# ---------------------------------------------------------------------------
# Merge + final shape
# ---------------------------------------------------------------------------

def merge_round1_features(account_features, tweet_features):
    """
    Left-join account features with tweet aggregates.

    Adds:
      - has_tweet_data: 1 if user appeared in tweet file, 0 otherwise.
        Crucial because some datasets (cresci_2017's tradition_spambots_*)
        ship users with no tweet data, which would otherwise become an
        all-zeros behaviour signature that correlates perfectly with
        the bot label.
      - log1p_tweet_count
      - tweets_per_day: from tweet_active_days when present, else falls
        back to account_age_days.

    Tweet behaviour columns are NaN (not 0) for users with no tweet data,
    so the model can distinguish "this user has zero retweets" from
    "we have no idea what this user does."
    """
    df = account_features.copy()
    tweet_features = tweet_features.copy()

    df["user_id"] = _canonicalize_id_series(df["user_id"])
    tweet_features["user_id"] = _canonicalize_id_series(tweet_features["user_id"])

    # has_tweet_data: known before the merge, so we can set it explicitly.
    users_with_tweets = set(tweet_features["user_id"].dropna())
    df["has_tweet_data"] = df["user_id"].isin(users_with_tweets).astype("Int8")

    merged = df.merge(tweet_features, on="user_id", how="left")

    # tweet_count_actual specifically: 0 is the right value when absent
    # (you tweeted nothing observed). Everything else stays NaN.
    if "tweet_count_actual" in merged.columns:
        merged["tweet_count_actual"] = merged["tweet_count_actual"].fillna(0)
    merged["log1p_tweet_count"] = np.log1p(merged["tweet_count_actual"].clip(lower=0))

    # tweets_per_day: prefer actual tweet timespan, fall back to account age.
    has_active = merged.get("tweet_active_days")
    if has_active is not None:
        rate_active = merged["tweet_count_actual"] / merged["tweet_active_days"]
    else:
        rate_active = pd.Series(np.nan, index=merged.index)

    rate_age = merged["tweet_count_actual"] / (merged["account_age_days"].clip(lower=0).fillna(0) + 1)
    merged["tweets_per_day"] = rate_active.where(rate_active.notna(), rate_age)

    return merged


def _resolve_availability(dataset_name, tweet_columns):
    """
    Resolve per-feature reliability for a dataset given which optional
    columns survived cleaning.

    The static `DATASET_FEATURE_AVAILABILITY` table assumes the worst case
    (e.g. twibot_2022 retweet detection from text is unreliable). If
    cleaning has been re-run with the new code, we now have canonical
    `is_retweet`/`is_reply` columns derived from the v2 referenced_tweets
    field. Detect those columns and upgrade the availability flag.
    """
    avail = dict(DATASET_FEATURE_AVAILABILITY.get(dataset_name, {}))

    if "is_retweet" in tweet_columns:
        avail["retweet_ratio"] = True
    if "is_reply" in tweet_columns:
        avail["reply_ratio"] = True
    if "created_at" in tweet_columns:
        avail["tweet_active_days"] = True

    return avail


def _apply_dataset_availability(df, dataset_name, resolved_availability=None):
    """Set to NaN any feature this dataset cannot support."""
    avail = (
        resolved_availability
        if resolved_availability is not None
        else DATASET_FEATURE_AVAILABILITY.get(dataset_name, {})
    )
    for col, available in avail.items():
        if not available and col in df.columns:
            df[col] = np.nan
    return df


def _ensure_canonical_schema(df, dataset_name):
    """
    Reindex to the canonical Round 1 schema so every dataset emits the same
    columns. Missing columns become NaN; extra columns are kept (so we don't
    lose `first_tweet_at` etc.) but moved after the canonical block.
    """
    df = df.copy()
    df["dataset"] = dataset_name

    # Make sure metadata columns exist
    for col in ROUND1_META_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    # Make sure every feature column exists
    for col in ROUND1_FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    canonical = ROUND1_META_COLS + ROUND1_FEATURE_COLS
    extras = [c for c in df.columns if c not in canonical]
    return df[canonical + extras]


def build_round1_features_for_dataset(dataset_name, paths):
    """
    End-to-end Round 1 build for a single dataset.

    `paths` must contain `users`, `tweets`, and `reference_date`.
    """
    users = read_users_csv(paths["users"])

    account_features = build_account_features(
        users_df=users,
        dataset_name=dataset_name,
        reference_date=paths["reference_date"],
    )

    tweet_columns = _tweet_columns(paths["tweets"])
    resolved_avail = _resolve_availability(dataset_name, tweet_columns)

    tweet_features = build_tweet_aggregates(paths["tweets"])
    merged = merge_round1_features(account_features, tweet_features)
    merged = _apply_dataset_availability(merged, dataset_name, resolved_avail)
    merged = _ensure_canonical_schema(merged, dataset_name)

    return merged


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_round1(df):
    """
    Sanity-check a combined-across-datasets Round 1 frame.

    Asserts the strict schema invariants the cross-dataset evaluation goal
    relies on. Returns a dict of dataset-level diagnostics and raises
    `AssertionError` on hard violations.
    """
    assert "dataset" in df.columns, "missing `dataset` column"
    assert "user_id" in df.columns, "missing `user_id` column"

    # Uniqueness of (dataset, user_id)
    dup = df.duplicated(subset=["dataset", "user_id"]).sum()
    assert dup == 0, f"{dup} duplicate (dataset, user_id) keys in Round 1 frame"

    # Every dataset has every Round 1 feature column
    missing = [c for c in ROUND1_FEATURE_COLS if c not in df.columns]
    assert not missing, f"missing Round 1 feature columns: {missing}"

    # Per-dataset variance check: a feature with zero variance inside a single
    # dataset is functionally a dataset-identity feature and will leak across
    # train/test eras.
    diagnostics = {}
    for dataset_name, g in df.groupby("dataset"):
        feat = g[ROUND1_FEATURE_COLS]
        nan_rate = feat.isna().mean().round(4)
        zero_var = []
        for col in ROUND1_FEATURE_COLS:
            s = pd.to_numeric(feat[col], errors="coerce").dropna()
            if len(s) > 1 and s.nunique() <= 1:
                zero_var.append(col)
        diagnostics[dataset_name] = {
            "rows": len(g),
            "nan_rate": nan_rate,
            "zero_variance_features": zero_var,
        }
    return diagnostics


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

    print("\nNaN rate per Round 1 feature, by dataset:")
    nan_table = (
        df.groupby("dataset")[feature_cols]
          .apply(lambda g: g.isna().mean().round(3))
    )
    print(nan_table.T.to_string())


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_feature_frame(df, out_path):
    """
    Save a feature dataframe.

    If `out_path` ends in `.parquet`, write Parquet (preferred — preserves
    dtypes and is much faster to load). Otherwise fall back to CSV.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Cross-era feature selection
# ---------------------------------------------------------------------------

# Features that are reference-date sensitive (raw counts and ages shift
# systematically across eras). Excluded from cross-era models by default;
# kept in the schema for within-dataset analysis. The log1p companions and
# per-day rates carry the same information in a reference-stable form.
CROSS_ERA_DENY_COLS = (
    "followers_count",
    "friends_count",
    "statuses_count",
    "listed_count",
    "account_age_days",
    "tweet_count_actual",
)

# Prefixes excluded from cross-era models. node2vec embeddings are computed
# independently per dataset on a sampled subgraph, so the spaces are not
# aligned across datasets and `emb_*` columns leak dataset identity.
CROSS_ERA_DENY_PREFIXES = ("emb_",)


def select_cross_era_features(
    df,
    candidate_cols=None,
    deny_cols=CROSS_ERA_DENY_COLS,
    deny_prefixes=CROSS_ERA_DENY_PREFIXES,
    max_nan_per_dataset=0.99,
    require_variance=True,
):
    """
    Return the subset of `candidate_cols` that is safe to use as features
    in a cross-dataset (cross-era) model.

    A column is kept iff:
      - it is not in `deny_cols`
      - it does not start with any prefix in `deny_prefixes`
      - in every dataset, its NaN rate is < `max_nan_per_dataset`
      - in every dataset, it has >1 unique non-NaN value (when `require_variance`)

    Returns:
        (kept_cols, excluded) where `excluded` is a list of (col, reason)
        tuples so you can audit what got dropped and why.
    """
    if candidate_cols is None:
        candidate_cols = [
            c for c in df.columns
            if c not in ROUND1_META_COLS and c != "label"
        ]

    assert "dataset" in df.columns, "frame is missing `dataset` column"

    kept = []
    excluded = []

    for col in candidate_cols:
        if col not in df.columns:
            excluded.append((col, "not in frame"))
            continue
        if col in deny_cols:
            excluded.append((col, "deny_cols (reference-date sensitive)"))
            continue
        if any(col.startswith(p) for p in deny_prefixes):
            excluded.append((col, "deny_prefixes (e.g. unaligned embeddings)"))
            continue

        per_ds_nan = df.groupby("dataset")[col].apply(lambda s: s.isna().mean())
        bad_ds = per_ds_nan[per_ds_nan >= max_nan_per_dataset]
        if len(bad_ds):
            excluded.append((
                col,
                f"NaN rate ≥ {max_nan_per_dataset} in: {bad_ds.index.tolist()}",
            ))
            continue

        if require_variance:
            per_ds_nunique = df.groupby("dataset")[col].apply(
                lambda s: pd.to_numeric(s, errors="coerce").dropna().nunique()
            )
            zero_var_ds = per_ds_nunique[per_ds_nunique <= 1]
            if len(zero_var_ds):
                excluded.append((
                    col,
                    f"zero variance in: {zero_var_ds.index.tolist()}",
                ))
                continue

        kept.append(col)

    return kept, excluded


def available_feature_cols(df, cols=ROUND1_FEATURE_COLS):
    """
    Strict version: returns columns from `cols` that are present in `df`,
    AND warns if any are missing (since the canonical schema is supposed
    to have all of them).
    """
    present = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        warnings.warn(
            f"Round 1 frame is missing canonical feature columns: {missing}. "
            f"Run `_ensure_canonical_schema` to materialize them as NaN.",
            stacklevel=2,
        )
    return present
