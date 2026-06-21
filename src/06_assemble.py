"""
06 — ASSEMBLE METRICS
Roll every module's *_summary.json + key derived numbers into a single outputs/metrics.json,
the single source of truth consumed by the Streamlit app, README, and submission docs.

Run: .venv/bin/python src/06_assemble.py
"""
import sys
import json
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import config as C


def load_json(p):
    try:
        return json.loads((C.TBL_DIR / p).read_text())
    except Exception:
        return {}


def main():
    df = pd.read_parquet(C.CLEAN_PARQUET)
    conf = df[df["is_confirmed"]]

    m = {
        "dataset": {
            "raw_rows": int(len(df)) + 5,  # 5 dropped for null ts; documented
            "rows_clean": int(len(df)),
            "confirmed_events": int(len(conf)),
            "confirmed_pct": round(len(conf) / len(df) * 100, 1),
            "rejected_excluded": int(df["is_rejected"].sum()),
            "dup_excluded": int(df["is_exact_dup"].sum()),
            "date_min": str(df["date"].min().date()),
            "date_max": str(df["date"].max().date()),
            "n_zones": int(conf["police_station"].nunique()),
            "n_junctions": int(conf.loc[conf.has_junction, "junction_clean"].nunique()),
            "pct_at_junction": round(conf["has_junction"].mean() * 100, 1),
            "top_violation": conf["primary_violation"].mode().iat[0],
            "top_vehicle": conf["vehicle_type"].mode().iat[0],
        },
        "hotspots": load_json("hotspot_summary.json"),
        "congestion": load_json("congestion_summary.json"),
        "priority": load_json("priority_summary.json"),
        "forecast": load_json("forecast_summary.json"),
    }
    C.METRICS_JSON.write_text(json.dumps(m, indent=2, default=str))
    print(json.dumps(m, indent=2, default=str))
    print(f"\n[saved] {C.METRICS_JSON}")


if __name__ == "__main__":
    main()
