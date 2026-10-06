# final_model: random forest + two-stage linear model

Everything needed to train, cross-validate and test the two final models, with the "show your work" material
(cross-validation, validation curves, leakage tests) and slide-ready figures. Forecast task: hourly RDU temperature,
12am Sep 17 to 11pm Sep 30 2026, using only data from before Sep 17.

Both models use **only RDU's own observations** (`cleaned_data/rdu_51_clean.csv`); no outside data is needed to run this folder.

## Where the code comes from, and what changed

| File | Source | Changes |
|---|---|---|
| `rf_model.py` | `random_forest.ipynb` (commit 854e0d9) | **One fix:** training forecasts start at local (Raleigh) midnight instead of UTC midnight. Features and notebook hyperparameters are otherwise unchanged. |
| `linear_model.py` | `linear_regression.ipynb` (commit 71c7270) | Logic copied as is. Data is passed in instead of being a global; added `predict_many`, `set_alpha` and a training-error helper for cross-validation. |
| `rdu_data.py` | new | Loads the RDU data and drops everything at/after the Sep 17 2026 cutoff. |

**The alignment fix.** The first push of the random forest built its training forecasts at midnight UTC (8pm Raleigh in September), but the real
forecast starts at midnight Raleigh time. In training, "lead hour 0" meant 8pm and the "last hour" features described the evening. The
linear model already used local midnight, so it is unaffected. **A fix was since pushed to the notebook itself (commit d29c7ed, merged into this branch):** it now works in
Raleigh time with local-midnight training forecasts, and its validation MAE dropped from 2.91 to 2.75, matching what `rf_model.py` gives on the same test (2.753).
`rf_model.py` here is the same fix, written so the original behavior can still be reproduced (`align="utc"`) for the comparison.

## Run order (from the repo root)

```bash
python -W ignore final_model/test_no_leakage.py     # must end with ALL LEAKAGE TESTS PASSED
python -W ignore final_model/cross_validation.py    # about 5 min -> results/cv_*.csv, results/chosen.json
python -W ignore final_model/make_figures.py        # -> figures/fig_cv_*.png
python -W ignore final_model/train_final.py         # about 5 min -> predictions/{rf,linear}_predictions.csv
bash experiments/fetch_final_truth.sh               # real Sep 17-30 observations (scoring only)
python -W ignore final_model/score_final.py         # run ONCE -> results/final_*.csv, figures/fig_final_*.png
```

## No data leakage

- `rdu_data.load()` clips every observation at/after the cutoff the moment the file is read.
- Every forecast uses only observations before its start; every training forecast has all 336 target hours before the training end
  (asserted in code for both models).
- `test_no_leakage.py` overwrites all data from a chosen time onward with garbage and checks that the features, training data, fitted model and
  forecast made at that time are unchanged. Run for Sep 17 2023, Sep 17 2025 and the real cutoff for both models. It also checks the garbage
  does change a model that is allowed to see it (so the test is not vacuous), and that the random forest's training forecasts all start at local midnight.
- The real Sep 17-30 observations are read only by `score_final.py`, which is run once after the models are fixed.

## Cross-validation (`cross_validation.py`)

4 time-ordered folds (2022-2025). Each fold trains only on forecasts whose 14-day window ends before Aug 15 of that year, then is scored on
21 forecasts (every 3 days, Aug 15 - Oct 15, issued at local midnight; 84 forecasts in total). Training error is measured on the same Aug 15 - Oct 15
season of the training years so it can be compared with validation error. Hyperparameters are chosen by lowest mean validation error.
Speed note: the random forest curves use 60 trees, 30% bootstrap and every 2nd training day; the final model uses the notebook's full settings.

| Model | MAE (°C), mean ± std over folds | RMSE (°C) |
|---|---|---|
| **Linear (two-stage), alpha=1000** | **2.76 ± 0.19** | 3.41 |
| Baseline: Fourier "normal" temperature only | 2.86 ± 0.23 | 3.49 |
| Baseline: same hour, ±7 days, prior years | 2.95 ± 0.28 | 3.64 |
| **Random forest (fixed), depth 6** | **3.07 ± 0.20** | 3.79 |
| Random forest (fixed), notebook depth 16 | 3.11 ± 0.27 | 3.86 |
| Random forest (original push: UTC-midnight training), depth 16 | 3.12 ± 0.26 | 3.86 |
| Baseline: repeat last 24 h | 3.82 ± 0.11 | 4.87 |

- **Random forest depth** (`figures/fig_cv_rf_depth.png`): training error keeps falling (3.95 to 1.44 °C) while validation error bottoms out at depth 6-8
  (3.07) and then creeps up (3.12 at depth 20). Modest overfitting; depth 6 chosen.
- **Ridge penalty** (`fig_cv_linear_alpha.png`): validation error is flat (2.76) from alpha 0.01 to 10,000, so the penalty barely matters; alpha=1000 has the lowest value by a hair.
  Training error 2.48 vs validation 2.76: a small, stable gap.
- **Feature groups** (`fig_cv_linear_groups.png`): temperature only 2.763, + dew point 2.777, + dew point + pressure 2.766. Extra groups do not help.
- **The alignment fix** (`fig_cv_rf_alignment.png`): depth 6: 3.063 (original push) vs 3.067 (fixed); depth 16: 3.116 vs 3.109. Essentially neutral in cross-validation.
  An earlier single-date test on the notebook's full settings (Sep 17 in 2022-2025) showed about 0.07 °C improvement. The fix is correct but its effect is small.

## Final test on the real Sep 17-30 2026 weather (`score_final.py`, run once)

The real data downloaded ends Sep 30 at 9:51am, so **322 of 336 hours** are scored. "Latest push" and "original push" are the random forest predictions files
at commits d29c7ed and 854e0d9.

| Model | MAE (°C) | MAE (°F) | RMSE (°C) | R² | Bias (°C) |
|---|---|---|---|---|---|
| Random forest (original push: UTC-midnight, depth 16) | 3.00 | 5.40 | 3.67 | 0.59 | -0.10 |
| Random forest (latest push: local-midnight fix, depth 16) | 3.00 | 5.40 | 3.64 | 0.60 | -0.62 |
| Linear (two-stage, as pushed) | 3.33 | 6.00 | 4.05 | 0.50 | -0.20 |
| **Linear (two-stage), this folder** | **3.35** | 6.03 | 4.07 | 0.49 | -0.20 |
| Average of this folder's two models | 3.47 | 6.24 | 4.24 | 0.45 | +0.37 |
| Baseline: same hour, ±7 days, prior years | 3.48 | 6.27 | 4.22 | 0.45 | +0.03 |
| Baseline: Fourier "normal" temperature only | 3.57 | 6.42 | 4.30 | 0.43 | -0.63 |
| **Random forest (alignment fix, CV-chosen depth 6), this folder** | **3.67** | 6.61 | 4.56 | 0.36 | +0.94 |
| Baseline: repeat last 24 h | 4.06 | 7.31 | 4.96 | 0.25 | +1.36 |

## Honest reading

1. **The linear port is faithful.** Cross-validation gives 2.76 (the notebook's own backtest: 2.77), and the final score is 3.35 vs 3.33 for the pushed predictions.
2. **Linear is the better model in cross-validation:** 2.76 vs 3.07 for the random forest, and the random forest does not beat the plain
   seasonal-normal baseline (2.86). The linear model's skill over that baseline is small (0.1 °C).
3. **The alignment fix is neutral**, in cross-validation (3.116 vs 3.109 at depth 16) and on the real window (the original and latest pushes both score 3.00).
4. **The depth-6 forest scored worse on the real window (3.67) than depth 16 (3.00).** Cross-validation could not separate them (3.07 vs 3.11, std about 0.2), so this
   difference comes from one 14-day spell (322 hours), not from the 84 forecasts in cross-validation. The depth-6 forest forecasts warm (+0.94 °C).
5. **Team decision needed:** keep the CV-chosen depth 6 (the principled choice, but the one that scored worst), or use the latest notebook as pushed (depth 16, chosen before any
   real-window scoring, and within noise of the best depth in CV). Either is defensible; if the pick is made after seeing these final scores, say so on the slides, because the real window is now a selection signal rather than a clean test.
6. Neither the linear model nor the forest clearly beats "same hour, ±7 days, prior years" (3.48) on the real window.

## Files

| Path | What |
|---|---|
| `rdu_data.py`, `rf_model.py`, `linear_model.py` | data loader and the two models |
| `test_no_leakage.py` | leakage tests |
| `cross_validation.py`, `make_figures.py` | cross-validation and its figures |
| `train_final.py`, `score_final.py` | final training and the one-time real-data scoring |
| `plot_style.py` | shared figure style (colorblind-checked palette) |
| `results/` | all tables: `cv_*.csv`, `chosen.json`, `final_metrics*.csv`, importances and coefficients |
| `figures/` | slide figures: `fig_cv_*` (validation curves, comparison, alignment, lead day) and `fig_final_*` |
| `predictions/` | the two final forecasts (`hour_local, predicted_temperature`) |
