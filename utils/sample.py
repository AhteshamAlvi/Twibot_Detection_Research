"""
sample.py

Draws a small sample from each pre-processed dataset (users + tweets)
and writes everything to a single CSV: data/samples/sample.csv

Columns present in only one file type (e.g. tweet_id) are NaN-filled
for rows that come from the other type.
"""

import os
from pathlib import Path
import pandas as pd

ROOT         = Path(__file__).resolve().parent.parent
PREPROCESSED = ROOT / "data" / "pre-processed"
OUT_PATH     = ROOT / "data" / "samples" / "sample.csv"
N            = 25   # rows per dataset per file type

DATASETS = {
    "cresci_2015": {
        "users":  PREPROCESSED / "cresci_2015" / "users_cresci_2015.csv",
        "tweets": PREPROCESSED / "cresci_2015" / "tweets_cresci_2015.csv",
    },
    "cresci_2017": {
        "users":  PREPROCESSED / "cresci_2017" / "users_cresci_2017.csv",
        "tweets": PREPROCESSED / "cresci_2017" / "tweets_cresci_2017.csv",
    },
    "twibot_2020": {
        "users":  PREPROCESSED / "twibot_2020" / "users_twibot_20.csv",
        "tweets": PREPROCESSED / "twibot_2020" / "tweets_twibot_20.csv",
    },
    "twibot_2022": {
        "users":  PREPROCESSED / "twibot_2022" / "users_twibot_22.csv",
        "tweets": PREPROCESSED / "twibot_2022" / "tweets_twibot_22.csv",
    },
}

frames = []

for dataset_name, paths in DATASETS.items():
    for file_type, path in paths.items():
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            continue

        df = pd.read_csv(path, low_memory=False, nrows=10_000)
        sample = df.sample(min(N, len(df)), random_state=42).copy()
        sample.insert(0, "file_type", file_type)
        sample.insert(0, "dataset",   dataset_name)
        frames.append(sample)
        print(f"  {dataset_name:<15} {file_type:<7} -> {len(sample)} rows, {len(df.columns)} cols")

combined = pd.concat(frames, ignore_index=True, sort=False)

os.makedirs(OUT_PATH.parent, exist_ok=True)
combined.to_csv(OUT_PATH, index=False)

print(f"\nSaved -> {OUT_PATH}")
print(f"Total rows : {len(combined)}")
print(f"Total cols : {len(combined.columns)}")
print(f"Columns    : {list(combined.columns)}")
