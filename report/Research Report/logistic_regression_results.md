# Logistic Regression Results

This summary captures the logistic regression baseline experiments for historical Twitter/X bot detection.

## Experimental setup

- Model: custom logistic regression implemented with `numpy` and trained with mini-batch gradient descent
- Features: 27 account/profile features derived from the processed user tables
- Evaluation split: stratified train/validation/test split with 70/10/20 proportions
- Class imbalance handling: balanced sample weights during training
- Important note: `twibot_2022` was run on a 100,000-row stratified sample to keep experimentation practical

## Test-set results

| Dataset | Rows used | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cresci_2015 | 5,301 | 0.8528 | 0.9372 | 0.8620 | 0.8980 | 0.9294 |
| cresci_2017 | 14,368 | 0.9826 | 0.9899 | 0.9872 | 0.9885 | 0.9952 |
| twibot_2020 | 11,826 | 0.8165 | 0.7588 | 0.9833 | 0.8566 | 0.8493 |
| twibot_2022 | 100,000 | 0.7613 | 0.3344 | 0.7124 | 0.4551 | 0.8190 |

## Observations

- `cresci_2015 (5,301 rows)`: accuracy was `0.8528` and ROC-AUC was `0.9294`. The most influential features were `account_age_days, has_bio, log_statuses`.
- `cresci_2017 (14,368 rows)`: accuracy was `0.9826` and ROC-AUC was `0.9952`. The most influential features were `log_favourites, log_statuses, log_friends`.
- `twibot_2020 (11,826 rows)`: accuracy was `0.8165` and ROC-AUC was `0.8493`. The most influential features were `verified_flag, bio_length, log_friends`.
- `twibot_2022 (100,000 rows)`: accuracy was `0.7613` and ROC-AUC was `0.8190`. The most influential features were `verified_flag, statuses_count, log_followers`.

## Interpretation

The logistic regression baseline performs very strongly on the Cresci datasets, moderately on TwiBot-20, and noticeably worse on the sampled TwiBot-22 data. This suggests that simple linear decision boundaries capture older bot patterns well, but the separation between bots and humans becomes weaker in newer datasets.

These results give the project a clean baseline for comparing against Random Forest and for evaluating whether bot characteristics drift over time.
