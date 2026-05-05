# Twibot Detection Research

Cross-era social-bot detection on Twitter / X. The project trains the same family of classifiers on each of four bot-detection datasets — **Cresci-2015**, **Cresci-2017**, **TwiBot-20**, and **TwiBot-22** — and then evaluates every trained model on every other dataset, producing a 4×4 transfer matrix per model type. The goal is to measure how the bot-detection problem itself has shifted across eras: a 2015-trained model that fails on 2022 data is evidence that bots have evolved, not just that one model is bad.

CMSC 396H research project, University of Maryland.

---

## What this repository contains

| Directory | Purpose |
|---|---|
| `notebooks/` | The full pipeline as numbered notebooks (00→04). |
| `utils/` | Pure-Python modules called from the notebooks. Feature builders, modeling helpers, figure utilities. |
| `data/raw/` | Original dataset releases. Not committed; download separately. |
| `data/pre-processed/` | Per-dataset cleaned `users_*.csv` and `tweets_*.csv` produced by notebook 00. |
| `data/processed/` | Engineered features + cached graph artefacts produced by notebook 02. |
| `data/models/` | Per-model-type pickle bundles produced by notebook 03 (one per train→test pair). |
| `report/` | LaTeX research report, research proposal, the literature folder, and `report/figures/` where every project figure is written. |
| `config/` | Reserved for centralized run config. |
| `requirements.txt` | Python dependencies. |

### Pipeline at a glance

```
data/raw/                  data/pre-processed/      data/processed/                  data/models/
   |                            |                        |                              |
   v                            v                        v                              v
[00_cleaning] -> users_*.csv ----+                 round1_base/    -+              {model_type}/
                tweets_*.csv     |                 round2_graph/    |              auc_matrix.csv
                                 v                 round2_full/     |              acc_matrix.csv
[01_eda]            ---> EDA tables and 1 figure   final_features/  +-> [03_ML]    results.json
                                                   edges/           |              <type>__train_X__test_Y.pkl
                                 |                                  |              ...
                                 v                                  v
                        [02_feature_engineering]            [04_visualizations] -> report/figures/04_viz__*.png
```

---

## Datasets

| Name | Year | Users (labeled) | Tweets | Bot rate | Label semantics | Graph |
|---|---:|---:|---:|---:|---|---|
| `cresci_2015` | 2015 | 5,301 | 2.8M | 75.2% | Curated subsets (TFP, TWT, FSF, INT, E13) | Full follow lists per subset |
| `cresci_2017` | 2017 | 14,368 | 6.6M | 75.8% | Mix of fake-followers + social/traditional spambots | Reply edges only |
| `twibot_2020` | 2020 | 11,826 | 2.0M | 55.7% | Annotator-labeled from a graph crawl | 1-hop neighbor field per user |
| `twibot_2022` | 2022 | 1,000,000 | 88.2M | 14.0% | Annotator-labeled from a graph crawl | Filtered follow + reply edges (~17.9M edges) |

**Datasets must be downloaded separately** and placed under `data/raw/` matching the structure in [notebooks/00_cleaning.ipynb](notebooks/00_cleaning.ipynb). The Cresci datasets are at <https://botometer.osome.iu.edu/bot-repository/datasets.html>; TwiBot-20 and TwiBot-22 require signing the dataset request form on their respective project pages.

---

## Quick start

```bash
# 1. Create venv + install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install duckdb networkx node2vec ijson scikit-learn xgboost matplotlib seaborn joblib pyarrow

# 2. Place raw datasets under data/raw/ (see notebooks/00_cleaning.ipynb for the expected layout)

# 3. Run notebooks in order
jupyter lab notebooks/00_cleaning.ipynb           # writes data/pre-processed/
jupyter lab notebooks/01_eda.ipynb                # exploratory tables + 1 figure
jupyter lab notebooks/02_feature_engineering.ipynb  # writes data/processed/
jupyter lab notebooks/03_ML_modeling.ipynb        # writes data/models/
jupyter lab notebooks/04_visualizations.ipynb     # writes report/figures/

# 4. Build the report
cd "report/Research Report"
pdflatex research_report.tex && bibtex research_report && pdflatex research_report.tex && pdflatex research_report.tex
```

`requirements.txt` only pins `numpy` and `pandas`; the FE pipeline additionally needs `duckdb`, `networkx`, `node2vec`, `ijson`, and the modeling pipeline needs `scikit-learn`, `xgboost`, `joblib`, `pyarrow`. These are not pinned by version yet — see Caveats.

---

## Notebook-by-notebook

### 00 — Cleaning

Reads the raw releases for each of the four datasets, normalizes user-profile columns into a common schema, and emits `data/pre-processed/{dataset}/users_{dataset}.csv` and `tweets_{dataset}.csv`. Per-dataset oddities (Cresci-2017's `L`-suffixed Unix-ms timestamps, TwiBot-20's plain-string tweets with no IDs or timestamps, TwiBot-22's v2 API JSON shards) are handled here so downstream code never has to special-case them.

The cleaning step also derives canonical `is_retweet` and `is_reply` boolean columns on each tweets file. For v1 datasets these come from text prefix and `in_reply_to_user_id` presence; for TwiBot-22 (v2 API), they come from the canonical `referenced_tweets` field. The FE step picks these up automatically when present and falls back to a heuristic when they're missing.

### 01 — EDA

Per-dataset summaries: row counts, label balance, column availability, missingness, account-metadata distributions, tweet-coverage heatmap. Produces `04_viz__01_eda__label_balance` (the only figure in this notebook).

### 02 — Feature engineering

Three rounds of feature construction:

1. **Round 1 (account + tweet behaviour)** — counts, log1p companions, profile-completeness flags, per-day rate features (`followers_per_day`, etc.), `account_age_days`, plus per-user tweet aggregates (text length, retweet ratio, reply ratio, hashtag/mention/URL averages, tweet timespan).
2. **Round 2 (graph features)** — degree, reciprocity, PageRank, neighbor degree aggregates, ego density, connected-component size. Edges are cached as `data/processed/edges/{dataset}_edges.parquet` after the first build.
3. **Round 3 (node2vec embeddings)** — per-dataset graph embeddings. Computed independently per dataset, so embedding spaces are NOT aligned across datasets and `emb_*` columns are excluded from the cross-era feature contract by default.

The notebook ends by computing two schema artefacts:
- `data/processed/final_features/feature_schema.json` — features safe across all four datasets simultaneously.
- `data/processed/final_features/pairwise_feature_schema.json` — features safe for each individual `(train, test)` dataset pair (16 entries: 4 within-dataset diagonals + 12 cross-era).

### 03 — ML modeling

For each of six classifier families — `logreg`, `ridge`, `lasso`, `rf`, `xgb`, `svm` — trains one full `sklearn.Pipeline` (impute → optionally scale → classify) per `(train_dataset, test_dataset)` pair using **that pair's** feature columns from the pairwise schema. 4×4 pairs × 6 model types = 96 pickled bundles per run.

Convention:
- **Off-diagonal pairs** (`train != test`) train on the full train set and evaluate on the full test set.
- **Diagonal pairs** (`train == test`) use a stratified 80/20 split inside the dataset, so the within-era AUC reflects held-out performance — not a training-set fit.

Each pickle is a self-describing `dict`: `{pipeline, feature_cols, train_dataset, test_dataset, model_type, metrics}`. Inference via `utils.modeling.predict_with_bundle(bundle, df)` validates that the dataframe carries every column the model was trained on, so feature-schema drift surfaces as a `ValueError` instead of silent column misalignment.

### 04 — Visualizations

Renders every figure for the report. Sections cover dataset structure, graph structure, per-model performance, cross-model comparison, feature importance, and the headline cross-era story plots. All figures save through `utils.figures.save_fig(name)` to `report/figures/{name}.png`.

---

## Key design decisions

### One canonical feature contract

`utils.base_features.ROUND1_FEATURE_COLS` and `utils.graph_features.GRAPH_FEATURE_COLS` together define the column set every dataset is reindexed against. Missing columns become explicit NaN per dataset rather than silently absent — this is what makes cross-dataset concatenation safe. `validate_round1(df)` asserts uniqueness of `(dataset, user_id)` and reports per-dataset NaN rates and zero-variance features so schema drift is caught at FE time, not silently inside a model fit.

### NaN means "we don't know," 0 means "we measured zero"

Throughout the FE pipeline, features that can't be computed for a given user (e.g., `pagerank` for a user sampled out of the PageRank computation, `tweet_active_days` when timestamps are missing, `reply_ratio` for TwiBot-20 which has no `in_reply_to_user_id`) stay as NaN. Features where 0 is meaningful (degree counts, reciprocity counts, component size) are zero-filled. This distinction is preserved end-to-end via Parquet output (CSV would coerce NaN to empty string and lose the dtype).

### Pairwise feature contract for cross-era models

Different `(train, test)` dataset pairs have different "safe" feature sets. TwiBot-20 has no `in_reply_to_user_id` so any pair involving it loses `reply_ratio`; Cresci-2017 has no follow graph so its pairs use a smaller graph-feature subset; etc. `select_cross_era_features()` resolves this per pair; the result is persisted as `pairwise_feature_schema.json` so the modeling code reads a contract instead of re-deriving the column list each run.

### String IDs everywhere

Every CSV read of a Twitter ID column passes `dtype={"user_id": str}`. Pandas would otherwise infer 18-19 digit IDs as `int64`, then upcast to `float64` on any NaN, then stringify as `"4.5e+18"`, then silently mismatch every join downstream. The string-ID rule is enforced in `utils.base_features.read_users_csv` and `utils.graph_features.clean_edges`.

### Reference-date-stable rate features

Raw counts and `account_age_days` shift systematically across eras (a 2022 human's follower count and account age are not on the same scale as a 2015 human's). The `*_per_day` rate features (`followers_per_day`, `friends_per_day`, `statuses_per_day`, `listed_per_day`, `tweets_per_day`) absorb that shift by dividing through `account_age_days + 1`. The cross-era feature selector excludes the raw counts by default and keeps the rates.

### Single figure-output convention

`utils.figures.save_fig(name)` writes to `report/figures/` from anywhere in the project, with a consistent dpi/format. Notebook prefix convention: `01_eda__*`, `02_fe__*`, `03_model__*`, `04_viz__*`. The LaTeX preamble sets `\graphicspath{{../figures/}}` so a figure is referenced by bare filename in the report — no path edits when figures are regenerated.

---

## Repository layout (annotated)

```
.
├── README.md                          this file
├── requirements.txt                   pip requirements (numpy, pandas — see Caveats)
├── config/                            reserved for run config
├── data/
│   ├── raw/                           original dataset releases (NOT committed)
│   │   ├── cresci-2015/
│   │   ├── cresci-2017/
│   │   ├── Twibot20/
│   │   └── Twibot22/
│   ├── pre-processed/                 cleaned per-dataset CSVs
│   │   └── {dataset}/users_*.csv, tweets_*.csv
│   ├── processed/                     engineered features + caches
│   │   ├── edges/                     cleaned edge lists (parquet, per dataset)
│   │   ├── round1_base/               account + tweet behavioural features
│   │   ├── round2_graph/              per-user graph features
│   │   ├── round2_full/               base + graph merged checkpoint
│   │   ├── round3_embeddings/         node2vec embeddings (auxiliary)
│   │   └── final_features/
│   │       ├── all_datasets_full_features.parquet
│   │       ├── feature_schema.json    cross-era contract (intersection of 4)
│   │       └── pairwise_feature_schema.json   per-pair contract (16 entries)
│   ├── models/                        trained model bundles
│   │   ├── logistic_regression/
│   │   ├── ridge/
│   │   ├── lasso/
│   │   ├── random_forest/
│   │   ├── xgboost/
│   │   └── linear_svm/
│   │       ├── auc_matrix.csv
│   │       ├── acc_matrix.csv
│   │       ├── results.json
│   │       └── {model_type}__train_{X}__test_{Y}.pkl   (16 per directory)
│   └── samples/                       small per-dataset sample for inspection
├── notebooks/
│   ├── 00_cleaning.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_ML_modeling.ipynb
│   └── 04_visualizations.ipynb
├── utils/
│   ├── cleaning.py                    drop_irrelevant_columns, extract_tweet_text, is_retweet/is_reply derivation
│   ├── base_features.py               Round 1 features, schema validation, cross-era selector
│   ├── graph_features.py              Round 2 features, edge loaders, node2vec wrapper
│   ├── modeling.py                    pairwise train/eval, save/load bundles, predict_with_bundle
│   ├── figures.py                     save_fig + standard color/dataset conventions
│   └── sample.py                      one-off inspection sample
└── report/
    ├── usenix2019_v3.sty              shared LaTeX style file
    ├── figures/                       all project figures land here (04_viz__*.png)
    ├── Literature/                    PDFs of cited papers
    ├── Research Proposal/             early proposal LaTeX source + PDF
    └── Research Report/
        ├── research_report.tex        the main report (framework + figure example)
        └── research_report.pdf        compiled output
```

---

## Reproducing the headline result

```bash
# Assumes raw datasets are already in data/raw/.

# Re-run the full pipeline end-to-end:
jupyter execute notebooks/00_cleaning.ipynb              # ~hours for TwiBot-22 cleaning
jupyter execute notebooks/02_feature_engineering.ipynb   # ~30min after edges cached
jupyter execute notebooks/03_ML_modeling.ipynb           # ~10min for all 6 models × 16 pairs
jupyter execute notebooks/04_visualizations.ipynb        # writes ~50 figures to report/figures/

# Build the report PDF:
cd "report/Research Report"
latexmk -pdf research_report.tex
```

The headline figure is `report/figures/04_viz__story__bots_over_time_logreg.png` (one line per training dataset, x-axis is the test dataset in chronological order). The 4×4 AUC matrix per model type lives at `data/models/{model_dir}/auc_matrix.csv`.

---

## Caveats and known gaps

- **`requirements.txt` is incomplete.** It lists only `numpy` and `pandas`. The full set of imports actually needed is documented in the Quick Start; pinning these by version is a TODO.
- **TwiBot-22 cleaning is slow** (~8–12 hours). Wall-clock dominated by the 9 JSON shards × 88M tweets streaming pass. Re-running cleaning is necessary if you want the canonical v2 retweet signal — without it, `retweet_ratio` for TwiBot-22 is NaN'd out by the static availability table.
- **Node2vec embeddings are not cross-dataset comparable.** They're computed independently per dataset on a sampled subgraph, so their spaces aren't aligned. Treat `emb_*` columns as within-dataset features only — `select_cross_era_features` drops them by default.
- **Modern (post-2022) data collection is not implemented.** The Twitter / X API has been heavily restricted and paywalled since 2023, so extending the time-series past TwiBot-22 is currently a blocker. The original plan included a `04_modern_data_collection` notebook; it was removed when the API access path became infeasible.
- **`account_age_days` uses each dataset's own reference date.** This is correct within a dataset (you can't extrapolate account age past the dataset's observation window) but means the raw value isn't comparable across datasets. The cross-era feature selector excludes `account_age_days` for that reason; use the `*_per_day` rate features for cross-era models.
- **`has_default_pic` and `verified` are 100% NaN in the Cresci datasets** because the raw releases don't include those fields. The cross-era selector drops them automatically.

---

## License and authorship

CMSC 396H research project at the University of Maryland.

**Authors:** Ahtesham Alvi (`aalvi1@terpmail.umd.edu`), Declan Scott, Michael Morton, Tristan Tang, Yudhiishbala Senthilkumar.

Dataset licenses and terms are governed by their respective providers — see the source dataset pages for redistribution and use restrictions. The Cresci, TwiBot-20, and TwiBot-22 datasets must be obtained separately and are not redistributed in this repository.
