"""Model / lessons slide: ECMWF's temperature bias at RDU drifted in mid-2025 (why MOS uses a recent-bias feature)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from style import BLUE, DATA, INK, MUTED, RED, footnote, save, title  # noqa: E402

b = pd.read_csv(DATA / "ecmwf_monthly_bias.csv")
x = range(len(b))
fig, ax = plt.subplots(figsize=(14, 6.2))
ax.bar(x, b["bias"], color=[BLUE if v < 0 else RED for v in b["bias"]], width=0.75, edgecolor="white", linewidth=2)
ax.axhline(0, color=INK, lw=1)
shift = b.index[b["month"] == "2025-08"][0]
ax.axvline(shift - 0.5, color=MUTED, lw=1.2)
ax.text(shift - 0.3, -1.85, "Aug 2025: the bias\nsuddenly shrinks", color=INK, fontsize=13, va="bottom")
ax.text(1, 0.32, "Below zero = ECMWF too cold", color=BLUE, fontsize=12.5)
ax.text(1, 0.55, "Above zero = ECMWF too warm", color=RED, fontsize=12.5)
ax.set_xticks(list(x)[::3])
ax.set_xticklabels([pd.Timestamp(m + "-01").strftime("%b\n%Y") for m in b["month"]][::3], fontsize=12.5)
ax.set_ylim(-2.2, 0.8)
ax.set_ylabel("ECMWF minus observed (°C)")
ax.grid(axis="x", visible=False)
title(ax, "The weather model's error at RDU is not constant",
      "A fixed correction learned before mid-2025 over-corrects newer runs, so MOS also uses ECMWF's recent 7- and 30-day error")
footnote(fig, "Monthly mean error of ECMWF 12 UTC runs at lead times up to 48 h. Sep 2026 covers Sep 1–16 only. "
              "Adding the recent-error feature cut the 2025 backtest error from 2.62 to 2.35 °C.")
save(fig, "fig5_ecmwf_bias_drift.png")
