"""
Run the full analysis pipeline end-to-end (modules 01 -> 06).
Each step writes its own tables/figures; 06 assembles outputs/metrics.json.

Run: .venv/bin/python src/run_pipeline.py
"""
import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    "01_clean.py",
    "02_hotspots.py",
    "03_congestion_proxy.py",
    "04_prioritization.py",
    "05_forecast.py",
    "06_assemble.py",
]


def main():
    t0 = time.time()
    for s in STEPS:
        print(f"\n{'#'*78}\n# RUNNING {s}\n{'#'*78}")
        t = time.time()
        runpy.run_path(str(HERE / s), run_name="__main__")
        print(f"--- {s} done in {time.time()-t:.1f}s ---")
    print(f"\n✅ pipeline complete in {time.time()-t0:.1f}s. See outputs/.")


if __name__ == "__main__":
    sys.exit(main())
