"""Leakage slide: what the forecast knew, and when (schematic timeline, no data needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from style import BLUE, GREY, INK, LIGHT, MUTED, ORANGE, RED, save  # noqa: E402

# x positions: Sep 16 6am .. Sep 17 12am mapped to 0..18 (hours); forecast window drawn as a block after a gap
H = lambda h: h - 6  # noqa: E731  hour of Sep 16 -> x
fig, ax = plt.subplots(figsize=(14, 6.2))
ax.set_xlim(-1, 33.6)
ax.set_ylim(-4.3, 6.1)
ax.axis("off")

ax.plot([H(6), H(24)], [0, 0], color=INK, lw=2.5, solid_capstyle="round")
ax.plot([H(24), H(25.5)], [0, 0], color=INK, lw=2.5, ls=(0, (1, 2)))
ax.add_patch(FancyBboxPatch((20.0, -0.7), 13.4, 1.4, boxstyle="round,pad=0,rounding_size=0.25", color=BLUE, alpha=0.18, lw=0))
ax.text(26.7, 0, "Forecast window\nSep 17 12am – Sep 30 11pm (336 hours)", ha="center", va="center",
        fontsize=13.5, color=INK, fontweight="bold")
for h, lab in [(6, "6am"), (12, "12pm"), (18, "6pm")]:
    ax.plot([H(h)] * 2, [-0.15, 0.15], color=INK, lw=1.5)
    ax.text(H(h), -0.45, lab, ha="center", va="top", color=MUTED, fontsize=12)
ax.text(H(6), 0.45, "Sep 16", ha="left", va="bottom", color=MUTED, fontsize=13, fontweight="bold")

events = [  # (hour on Sep 16, y of label, color, text)
    (8, 2.2, ORANGE, "8am: ECMWF run starts\n(12 UTC run on Sep 16)"),
    (13, 3.3, ORANGE, "~12–2pm: run published\n(we later download it from\nthe Open-Meteo archive)"),
    (23.85, 2.2, GREY, "11:51pm: last RDU\nobservation used"),
    (24, -1.9, RED, "12am Sep 17: CUTOFF\nforecast issued, model frozen"),
]
for h, y, col, txt in events:
    ax.plot([H(h)] * 2, [0, y * 0.72], color=col, lw=1.5)
    ax.scatter([H(h)], [0], s=110, color=col, zorder=3, edgecolor="white", linewidth=2)
    ax.text(H(h), y, txt, ha="center", va="center", fontsize=12.5, color=INK)

ax.add_patch(FancyBboxPatch((20.0, -4.2), 13.4, 1.6, boxstyle="round,pad=0,rounding_size=0.25", color=LIGHT, lw=0))
ax.text(26.7, -3.4, "Real Sep 17–30 temperatures:\ndownloaded after all models were frozen,\nused once, only for scoring",
        ha="center", va="center", fontsize=12.5, color=INK)

ax.text(-1, 6.05, "What our forecast knew, and when", fontsize=19, fontweight="bold", ha="left", va="top", color=INK)
ax.text(-1, 5.35, "Every input existed before 12am Sep 17. Checked by asserts and tests that overwrite later data.",
        fontsize=14, ha="left", va="top", color=MUTED)
ax.text(-1, -2.75, "Other guards:\n• training forecasts only use targets before each test season\n"
                  "• observations clipped at the cutoff when loaded\n• features use only hours before the forecast",
        fontsize=12.5, ha="left", va="top", color=INK, linespacing=1.5)
save(fig, "fig6_leakage_timeline.png", bottom=0)
