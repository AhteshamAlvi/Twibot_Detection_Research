import pandas as pd


def compute_derived_features(df, reference_date):
    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    ref = pd.Timestamp(reference_date, tz="UTC")
    df["account_age_days"] = (ref - df["created_at"]).dt.days
    df["ff_ratio"]         = df["followers_count"] / (df["friends_count"] + 1)
    df["has_bio"]          = df["description"].notna() & (df["description"].str.strip() != "")
    df["has_location"]     = df["location"].notna()    & (df["location"].str.strip() != "")
    df["has_default_pic"]  = df["default_profile_image"].astype(str).str.strip() == "1"
    df = df.drop(columns=["description", "location", "default_profile_image"])
    print("Derived features added: account_age_days, ff_ratio, has_bio, has_location, has_default_pic")
    print(f"Null account ages (unparseable dates): {df['account_age_days'].isna().sum()}")
    return df


def aggregate_tweets(tweets_df):
    tweets_df = tweets_df.copy()
    tweets_df["is_retweet"]    = tweets_df["text"].str.startswith("RT @")
    tweets_df["hashtag_count"] = tweets_df["text"].str.count(r"#\w+")
    tweets_df["mention_count"] = tweets_df["text"].str.count(r"@\w+")
    tweets_df["url_count"]     = tweets_df["text"].str.count(r"http\S+")
    agg = tweets_df.groupby("user_id").agg(
        tweet_count_actual=("tweet_id",       "count"),
        retweet_ratio     =("is_retweet",     "mean"),
        avg_hashtags      =("hashtag_count",  "mean"),
        avg_mentions      =("mention_count",  "mean"),
        avg_urls          =("url_count",      "mean"),
    ).reset_index()
    return agg


def merge_tweet_features(users_df, agg_df):
    users_df = users_df.copy()
    agg_df   = agg_df.copy()
    users_df["user_id"] = users_df["user_id"].astype(str)
    agg_df["user_id"]   = agg_df["user_id"].astype(str)
    merged = users_df.merge(agg_df, on="user_id", how="left")
    print(f"Rows before merge: {len(users_df)}")
    print(f"Rows after merge:  {len(merged)}")
    print(f"Users missing tweet features: {merged['tweet_count_actual'].isna().sum()}")
    return merged
