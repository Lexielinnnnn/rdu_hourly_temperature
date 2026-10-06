"""Run every presentation figure script (from the repo root):  python presentation/make_figures.py

Needs presentation/data/ (committed; rebuild with presentation/make_data.py).  Output: presentation/figures/*.png
"""
import runpy
from pathlib import Path

for script in sorted(Path(__file__).resolve().parent.glob("fig*.py")):
    runpy.run_path(str(script), run_name="__main__")
