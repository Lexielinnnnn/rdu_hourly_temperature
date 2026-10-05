# Sam's work: data sources, features, leakage protection (branch `sam-teshome`)

Project 1: predict hourly RDU temperature for 12am Sep 17 to 11pm Sep 30 2026 using only data from before Sep 17.
Required: 1 linear regression + 1 other model. Full plan, numbers and task split are in `PLAN.md`;
the teammate summary is `Project1_Action_Plan.docx`.

## What I did

1. **Chose the models.** Ridge linear regression (required) and a KNN regressor (k=300, small feature set) as the second
   model, after reviewing the 520 midterm material. The random forest already in the repo stays as an extra comparison.
2. **Added alternative data** beyond the RDU files, all downloaded by `experiments/fetch_alt_data.sh`:
   - corridor stations (NOAA GHCNh): DC, Greensboro, Charlotte, Roanoke, Norfolk, Charleston WV, Wilmington NC
   - mid-Atlantic 100 m winds and sea-level pressure at 5 points (ERA5 via Open-Meteo)
   - soil moisture and cloud cover at 3 points (ERA5-Land via Open-Meteo)
   - daily sea-surface temperature at 4 points (NOAA OISST)
   - NAO, AO, PNA (daily) and Nino 3.4 (monthly) climate indices (NOAA CPC)
   - year-prior feature: RDU at the target hour 1 and 2 years earlier (built from existing data)
3. **Built the feature pipeline** (`experiments/pipeline.py`) with a hard cutoff at 12am Sep 17 2026.
4. **Proved there is no leakage** (`experiments/test_no_leakage.py`).
5. **Backtested** the task on Sep 17-30 of 2022-2025 (`experiments/backtest.py`), one data group at a time.

## Leakage protection

- Every source is clipped to strictly before the cutoff on load (the 2026 station files and CPC files run past Sep 30).
- Daily and monthly sources are delayed by a publication lag (2 days / 2 months).
- Every feature is a trailing window ending 1 hour before the forecast origin.
- The test overwrites all data from a chosen time onward with garbage and checks features and training rows do not change
  (real forecast origin plus the 2023 and 2025 test origins). It also checks the garbage does change later features.
- The backtest asserts no training target reaches the test origin.
- The real Sep 17-30 RDU observations are NOT in the repo's data. They are downloaded separately by `experiments/fetch_final_truth.sh` into `raw_data/final_truth/` and read only by `experiments/score_final.py` (final scoring, run once).

## Results (mean absolute error in C, average of Sep 17-30 in 2022-2025)

| Model / features | MAE | Day 1 |
|---|---|---|
| Baseline: same hour last 2 years | 3.12 | 3.35 |
| Baseline: repeat last 24 h | 3.93 | 2.28 |
| Ridge, RDU + year-prior | 3.12 | 2.10 |
| Ridge, + soil/cloud | 3.02 | 1.96 |
| Ridge, + wind | 3.03 | 2.25 |
| KNN k=300, compact | 3.12 | 2.75 |
| KNN, compact + DC temperature | 3.03 | 2.74 |
| Average of ridge + KNN | 3.05 | 2.23 |
| Ridge, all groups at once | 3.44 | 2.02 |

Takeaways:
- Gains from the new data are small (about 0.1 C) and within year-to-year variation (2.0 to 4.4 C), so treat them as suggestive.
- Adding every group at once hurts; add groups one at a time.
- Short range (day 1) benefits most; days 4-14 stay near 3.2-3.4 C for every model.

## How to run (from the repo root)

```bash
bash experiments/fetch_alt_data.sh                  # about 650 MB into raw_data/ (gitignored)
python -W ignore experiments/test_no_leakage.py     # must end with ALL LEAKAGE TESTS PASSED
python -W ignore experiments/backtest.py            # writes experiments/results/backtest_results.csv
```

Needs pandas, numpy and scikit-learn.

## Files

| File | Purpose |
|---|---|
| `experiments/fetch_alt_data.sh` | download all alternative data |
| `experiments/pipeline.py` | load sources, enforce cutoff, build features |
| `experiments/test_no_leakage.py` | leakage tests |
| `experiments/backtest.py` | early feature-group backtest (Sep 17 only) |
| `experiments/modeling.py` | shared features, models and folds |
| `experiments/model_selection.py` | cross-validation and validation curves (train vs validation error) |
| `experiments/make_cv_figures.py`, `plot_style.py` | slide figures for the CV results |
| `experiments/final_forecast.py` | train chosen models on data before Sep 17, write `predictions_*.csv` |
| `experiments/fetch_final_truth.sh`, `score_final.py` | real observations and one-time final scoring |
| `figures/`, `results/` | all slide figures and metric tables |
| `experiments/results/backtest_results.csv` | backtest output |
| `PLAN.md` | data rationale, results, caveats, task split |
| `Project1_Action_Plan.docx` | teammate action plan |

## Model selection and final test (Mon Oct 5)

**Cross-validation** (`experiments/model_selection.py`): 4 time-ordered folds (2022-2025). Each fold trains only on forecast dates whose
14-day window ends before Aug 15 of that year, then scores 21 forecasts (every 3 days, Aug 15 - Oct 15). Training error is measured on the
same Aug 15 - Oct 15 season of the training years so it is comparable with validation error. Figures are in `figures/`, tables in `results/`.

| Model (tuned) | Chosen setting | CV MAE (C) | CV RMSE (C) |
|---|---|---|---|
| KNN | k=4000, compact features | 2.91 | 3.62 |
| Ridge linear regression | RDU + year-prior + ocean temp, alpha=10000 | 2.95 | 3.62 |
| Random forest | max depth 6 | 3.04 | 3.79 |
| Baseline: same hour, last 2 yrs | | 3.13 | 3.87 |
| Baseline: repeat last 24 h | | 3.81 | 4.89 |

- **Other model = KNN**, since it beat the random forest in CV. Differences between KNN and ridge are within the fold-to-fold spread (std about 0.2 C).
- **Overfitting check** (`fig_cv_rf_depth.png`, `fig_cv_knn_k.png`, `fig_cv_ridge_alpha.png`): the random forest overfits past depth 6
  (training error keeps falling from 3.6 to 1.3 C while validation error rises), so depth 6 was chosen. KNN's validation error
  flattens from k=2000 to 4000. The ridge penalty barely matters: the curve is flat until alpha = 10^4 because there are about 500k training rows.
- Feature-group choice for ridge (`fig_cv_ridge_feature_groups.png`) is a near tie (3.03-3.07 C for most groups, ocean temp 2.99), so treat it as weak evidence.

**Final test** (`experiments/final_forecast.py`, then `experiments/score_final.py`, run once): models retrained on all data before Sep 17 2026
and scored on the real Sep 17-30 hours. The real data downloaded ends Sep 30 at 9:51am, so 322 of 336 hours are scored.

| Model | MAE (C) | RMSE (C) | R2 | Bias (C) |
|---|---|---|---|---|
| Lexie's random forest (as pushed) | 3.00 | 3.67 | 0.59 | -0.10 |
| Burak's linear model (as pushed) | 3.33 | 4.05 | 0.50 | -0.20 |
| Ridge linear regression (ours) | 3.69 | 4.56 | 0.36 | +0.89 |
| KNN (ours) | 3.72 | 4.57 | 0.36 | +0.71 |
| Baseline: repeat last 24 h | 4.06 | 4.96 | 0.25 | +1.36 |
| Random forest (ours) | 4.13 | 5.07 | 0.21 | +1.26 |
| Baseline: same hour, last 2 yrs | 4.59 | 5.61 | 0.04 | +1.63 |

**Honest read of the final test:** our two models beat both baselines but score worse than the teammates' models as pushed.
This is one 14-day window, and the weather was unusual (a heat wave to about Sep 21, then a cool spell Sep 22-26 that none of our smooth
forecasts anticipated; see `fig_final_forecast_vs_actual.png`). All three of our models forecast too warm on average, while the teammates' models do not.
The teammates' models were not run through our cross-validation, so the CV table and the final table are not directly comparable.
Do not tune on the final window; choose the submitted models with a shared CV protocol (open item below).

## Reproduce everything (from the repo root)

```bash
bash experiments/fetch_alt_data.sh                  # alternative data (about 650 MB, gitignored)
python -W ignore experiments/test_no_leakage.py     # must end with ALL LEAKAGE TESTS PASSED
python -W ignore experiments/model_selection.py     # CV + validation curves, about 10 min -> results/
python -W ignore experiments/make_cv_figures.py     # -> figures/fig_cv_*.png
python -W ignore experiments/final_forecast.py      # -> predictions_{ridge,knn,rf}.csv
bash experiments/fetch_final_truth.sh               # real Sep 17-30 observations, for scoring only
python -W ignore experiments/score_final.py         # run ONCE -> results/final_*.csv, figures/fig_final_*.png
```

## Action plan and division of labor (due Wed Oct 7)

| | Sam: data, features, KNN | Burak: linear regression, evaluation | Lexie: cleaning, RF, writeup, slides |
|---|---|---|---|
| **Mon Oct 5** | Make `pipeline.py` the single source of data and features (move to `src/`, keep the leakage test passing). Tune KNN (k, feature subset) with time-ordered CV over many forecast dates, not just Sep 17. | Final ridge / lasso / OLS comparison; LassoCV to see which feature groups survive. Collinearity and residual checks (Week 3). | Fix `clean_data.ipynb` (nearest-:51 rule, fill gaps) and re-export the cleaned RDU file. Rerun `random_forest.ipynb` as an extra model. Start the writeup outline. |
| **Tue Oct 6** | Feature-group ablation table; test different features for short vs long lead. Final KNN predictions to `predictions_knn.csv`. | Scoring script: all models on the real Sep 17-30 data (MAE, RMSE, R2) vs the two baselines, plus per-day error plot. Final predictions to `predictions_linreg.csv`. | Build the slide deck and merge everyone's writeup sections. Collect figures from Sam and Burak. |
| **Wed Oct 7** | Clean run from a fresh clone; README with run order. | Check slide numbers match the results table. | Proofread writeup and slides; submit. |

**Slides** (required: inputs, pipeline, features, models, evaluation approach, performance):
- Lexie: title and problem, data pipeline and cleaning, random forest (extra model), what did not work, conclusion.
- Sam: inputs (RDU plus the alternative data groups and why each), feature design, leakage protection, KNN model.
- Burak: linear regression, evaluation approach (backtest on Sep 17-30 of 2022-2025, baselines), performance on the real Sep 17-30 2026.

**Writeup (2-4 pages):** each person writes their own section in their own words (Sam: data and features; Burak: linear model and evaluation; Lexie: introduction, cleaning, how we applied course ideas). Lexie merges and edits on Tuesday. AI is allowed for code only; writing and slides must be our own.

**Open questions for Monday:** Celsius or Fahrenheit for the grader? Do weather-model forecasts issued before Sep 17 count as allowed data?

## Known caveats and not done

- `clean_data.ipynb` keeps only `:51` observations; KDCA reports at `:52` and 2026 files are sub-hourly, so exact-minute
  filtering drops hours. `pipeline.py` uses the report nearest `:51`.
- Cleaned RDU data is in Celsius; confirm what the grader wants.
- Not done: hurricane tracks, MJO, upper-air heights, the advection/ring features (idea only), a shared CV protocol that also runs the teammates' models.
- Open question: whether weather-model forecasts issued before Sep 17 count as allowed data.
