"""
Cleaning helpers used by `notebooks/00_cleaning.ipynb`.

The new contract: every pre-processed `tweets_*.csv` should have, where
the source data supports it, the canonical columns:
    tweet_id, user_id, text, created_at, in_reply_to_user_id,
    is_retweet, is_reply

`is_retweet` and `is_reply` are derived once at cleaning time from the
source-appropriate signal:
    - cresci_2015 / cresci_2017 / twibot_2020 (v1 API tweets):
        is_retweet := text starts with "RT @"
        is_reply   := in_reply_to_user_id is non-empty (cresci_2015/2017 only)
    - twibot_2022 (v2 API):
        is_retweet := referenced_tweets contains a {type: "retweeted", ...}
        is_reply   := referenced_tweets contains a {type: "replied_to", ...}
                      OR in_reply_to_user_id is non-empty

Computing these at cleaning time means the FE step doesn't have to encode
a per-dataset per-API-version detection rule into a SQL aggregate.
"""

import pandas as pd


def drop_irrelevant_columns(df, keep_cols):
    dropped = [c for c in df.columns if c not in keep_cols]
    df = df[[c for c in keep_cols if c in df.columns]].copy()
    print(f"Kept {df.shape[1]} columns, dropped {len(dropped)}")
    print(f"Dropped: {dropped}")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    print(f"Remaining nulls:\n{nulls.to_string() if len(nulls) else '  none'}")
    return df


def derive_is_retweet_from_text(text_series):
    """v1 retweet heuristic: text starts with 'RT @'."""
    return (
        text_series.fillna("")
        .astype(str)
        .str.startswith("RT @")
        .astype("Int8")
    )


def derive_is_reply_from_reply_id(reply_id_series):
    """A reply is a tweet whose in_reply_to_user_id field is set."""
    s = reply_id_series.astype("string").str.strip()
    is_reply = s.notna() & ~s.isin(["", "0", "nan", "None"])
    return is_reply.astype("Int8")


def derive_v2_flags_from_referenced_tweets(referenced_tweets):
    """
    For Twitter API v2 records, retweets and replies are encoded inside the
    `referenced_tweets` field as a list of {type, id} dicts. Returns
    (is_retweet, is_reply) as 0/1 ints.
    """
    if not referenced_tweets:
        return 0, 0
    types = {ref.get("type") for ref in referenced_tweets if isinstance(ref, dict)}
    return int("retweeted" in types), int("replied_to" in types)


def extract_tweet_text(tweets_df, labeled_user_ids,
                       tweet_id_col="id", user_id_col="user_id",
                       text_col="text", created_col="created_at",
                       reply_id_col="in_reply_to_user_id",
                       extra_cols=None):
    """
    Filter raw tweets to labeled users, normalize column names, and add
    canonical `is_retweet` / `is_reply` flags derived from text and
    `in_reply_to_user_id` when present.
    """
    tweets_df = tweets_df.copy()
    tweets_df[user_id_col] = tweets_df[user_id_col].astype(str)
    labeled_user_ids = set(str(i) for i in labeled_user_ids)

    filtered = tweets_df[tweets_df[user_id_col].isin(labeled_user_ids)].copy()

    keep = [tweet_id_col, user_id_col, text_col, created_col]
    if reply_id_col in filtered.columns and reply_id_col not in keep:
        keep.append(reply_id_col)
    if extra_cols:
        keep += [c for c in extra_cols if c in filtered.columns and c not in keep]

    extracted = filtered[[c for c in keep if c in filtered.columns]].copy()
    extracted = extracted.rename(columns={
        tweet_id_col: "tweet_id",
        user_id_col:  "user_id",
        text_col:     "text",
        created_col:  "created_at",
        reply_id_col: "in_reply_to_user_id",
    })

    if "tweet_id" in extracted.columns:
        extracted["tweet_id"] = extracted["tweet_id"].astype(str)

    # is_retweet from text prefix (v1 heuristic — works for cresci + twibot_2020)
    if "text" in extracted.columns:
        extracted["is_retweet"] = derive_is_retweet_from_text(extracted["text"])

    # is_reply from in_reply_to_user_id presence
    if "in_reply_to_user_id" in extracted.columns:
        extracted["is_reply"] = derive_is_reply_from_reply_id(
            extracted["in_reply_to_user_id"]
        )

    print(f"Tweets from labeled users: {len(extracted):,} / {len(tweets_df):,} total")
    return extracted
