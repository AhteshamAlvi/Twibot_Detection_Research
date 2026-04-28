import pandas as pd


def drop_irrelevant_columns(df, keep_cols):
    dropped = [c for c in df.columns if c not in keep_cols]
    df = df[keep_cols].copy()
    print(f"Kept {len(keep_cols)} columns, dropped {len(dropped)}")
    print(f"Dropped: {dropped}")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    print(f"Remaining nulls:\n{nulls.to_string() if len(nulls) else '  none'}")
    return df


def extract_tweet_text(tweets_df, labeled_user_ids,
                       tweet_id_col="id", user_id_col="user_id",
                       text_col="text", created_col="created_at"):
    tweets_df = tweets_df.copy()
    tweets_df[user_id_col] = tweets_df[user_id_col].astype(str)
    labeled_user_ids = set(str(i) for i in labeled_user_ids)

    filtered = tweets_df[tweets_df[user_id_col].isin(labeled_user_ids)]
    extracted = filtered[[tweet_id_col, user_id_col, text_col, created_col]].copy()
    extracted = extracted.rename(columns={
        tweet_id_col: "tweet_id",
        user_id_col:  "user_id",
        text_col:     "text",
        created_col:  "created_at",
    })
    extracted["tweet_id"] = extracted["tweet_id"].astype(str)
    print(f"Tweets from labeled users: {len(extracted):,} / {len(tweets_df):,} total")
    return extracted
