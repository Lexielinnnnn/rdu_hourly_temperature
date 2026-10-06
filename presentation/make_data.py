"""Build the small CSV tables the presentation figures read (run once, ~4 min; figures then run in seconds).

    python -W ignore presentation/make_data.py        # from the repo root

Writes presentation/data/:
  mos_backtest.csv          MOS backtest (42 forecasts, Aug 15 - Oct 15, 2024 + 2025), per forecast x lead day x model
  mos_coefficients.csv      coefficients of the final MOS model (trained on everything before Sep 17 2026)
  ecmwf_monthly_bias.csv    ECMWF - observed at RDU, by month (leads <= 48 h)
  final_window.csv          every model's Sep 17-30 2026 forecast next to the real temperatures
  final_metrics_all.csv     MAE / RMSE / R2 / bias of every model on the real window
  normal_vs_actual_2026.csv observed temperature vs the climatology "normal", Aug 1 - Sep 30 2026

No model is changed or re-tuned here.  The real Sep 17-30 temperatures come from
final_model/results/final_forecast_vs_actual.csv (the one-time scoring file) and are only compared against forecasts
that were frozen before scoring (the committed prediction files).  The script asserts that the MOS model rebuilt
from final_model/mos_model.py reproduces the frozen rdu_mos_predictions.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "final_model"))
import mos_model as mm  # noqa: E402

OUT = Path("presentation/data")
OUT.mkdir(parents=True, exist_ok=True)
TZ, HORIZON, LEAD = mm.TZ, mm.HORIZON, mm.LEAD

D = mm.Data()

# ---------------------------------------------------------------- 1. MOS backtest (same protocol as nwp_mos.ipynb)
rows = []


def score(pred, origin, model):
    actual = D.temp_local.reindex(pd.date_range(origin, periods=HORIZON, freq="h", tz=TZ)).values
    err = pred - actual
    for d in range(14):
        e = err[d * 24:(d + 1) * 24]
        e = e[~np.isnan(e)]
        rows.append({"origin": origin, "model": model, "lead_day": d + 1,
                     "abs_err": np.abs(e).sum(), "sq_err": (e ** 2).sum(), "n": len(e)})


for year in [2024, 2025]:
    origins = pd.date_range(f"{year}-08-15", f"{year}-10-15", freq="3D", tz=TZ)
    m = mm.MOS(D).fit(origins[0])
    print(f"{year}: trained on {m.n_runs} ECMWF runs")
    for o in origins:
        assert D.run_for(o) <= o.tz_convert("UTC") - mm.MIN_RUN_AGE
        hours = o.tz_convert("UTC") + pd.to_timedelta(LEAD, unit="h")
        score(m.predict(o), o, "MOS (ECMWF + station)")
        score(m.raw_nwp(o), o, "Raw ECMWF")
        score(m.fallback.predict(o), o, "Station-only linear")
        score(m.clim.predict(hours), o, "Climatology")

bt = pd.DataFrame(rows)
bt.to_csv(OUT / "mos_backtest.csv", index=False)
per = bt.groupby(["model", "origin"])[["abs_err", "n"]].sum()
summary = (per.abs_err / per.n).groupby("model").mean().round(2)
print("Backtest MAE:", summary.to_dict())
assert abs(summary["MOS (ECMWF + station)"] - 2.24) < 0.005, "MOS backtest no longer matches the notebook (2.24)"

# ---------------------------------------------------------------- 2. final model: reproduce frozen forecast, coefficients
final = mm.MOS(D).fit(mm.CUTOFF)
pred = final.predict(mm.CUTOFF)
frozen = pd.read_csv("rdu_mos_predictions.csv")["predicted_temperature"].to_numpy()
assert np.allclose(pred, frozen, atol=1e-6), "mos_model.py does not reproduce rdu_mos_predictions.csv"
print("Frozen MOS forecast reproduced exactly.")
final.coefficients().to_csv(OUT / "mos_coefficients.csv", index=False)

# ---------------------------------------------------------------- 3. ECMWF bias by month
eb = D.err_by_hour.dropna()
month = eb.index.tz_convert(TZ).strftime("%Y-%m")
pd.DataFrame({"bias": eb.groupby(month).mean(), "n_hours": eb.groupby(month).size()}).to_csv(
    OUT / "ecmwf_monthly_bias.csv", index_label="month")

# ---------------------------------------------------------------- 4. every model on the real Sep 17-30 window
hours = pd.date_range(mm.CUTOFF, periods=HORIZON, freq="h", tz=TZ)
key = hours.tz_convert("UTC")


def by_hour(path, col, index_col=None):
    d = pd.read_csv(path, index_col=index_col)
    idx = pd.to_datetime(d.index if index_col is not None else d["hour_local"], utc=True)
    return pd.Series(d[col].to_numpy(), index=idx).reindex(key).to_numpy()


fva = "final_model/results/final_forecast_vs_actual.csv"
sam = "results/final_forecast_vs_actual.csv"
win = pd.DataFrame({
    "actual": by_hour(fva, "actual", 0),
    "MOS (ECMWF + station)": frozen,
    "Raw ECMWF": D.nwp_path(mm.CUTOFF, LEAD),
    "Raw ECMWF + climatology after day 11": final.raw_nwp(mm.CUTOFF),
    "Random forest": by_hour(fva, "Random forest (Lexie, latest push)", 0),
    "Station-only linear": by_hour("rdu_linear_predictions.csv", "predicted_temperature"),
    "Ridge + alternative data": by_hour(sam, "Ridge linear regression (ours)", 0),
    "KNN analogs": by_hour(sam, "KNN (ours)", 0),
    "Baseline: same hour, prior years": by_hour(fva, "Baseline: same hour, +/-7 days of prior years", 0),
    "Baseline: climatology (normal)": final.clim.predict(key),
    "Baseline: repeat last 24 h": by_hour(fva, "Baseline: repeat last 24 h", 0),
}, index=hours)
win.index.name = "hour_local"
win.to_csv(OUT / "final_window.csv")

scored = win[win["actual"].notna()]
y = scored["actual"]
metrics = []
for c in win.columns.drop(["actual", "Raw ECMWF"]):
    e = scored[c] - y
    metrics.append({"model": c, "MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()),
                    "R2": 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(), "bias": e.mean(), "n_hours": e.notna().sum()})
metrics = pd.DataFrame(metrics).sort_values("MAE")
metrics.to_csv(OUT / "final_metrics_all.csv", index=False)
print(metrics.round(2).to_string(index=False))

# ---------------------------------------------------------------- 5. observed vs normal, Aug 1 - Sep 30 2026
span = pd.date_range("2026-08-01", "2026-10-01", freq="h", tz=TZ, inclusive="left")
obs_before = D.temp_local.reindex(span)
obs_after = pd.Series(win["actual"].to_numpy(), index=hours).reindex(span)
pd.DataFrame({"observed": obs_before.fillna(obs_after).to_numpy(),
              "normal": final.clim.predict(span.tz_convert("UTC")),
              "after_cutoff": span >= mm.CUTOFF}, index=span).to_csv(OUT / "normal_vs_actual_2026.csv",
                                                                       index_label="hour_local")
print("Wrote", sorted(p.name for p in OUT.glob("*.csv")))
