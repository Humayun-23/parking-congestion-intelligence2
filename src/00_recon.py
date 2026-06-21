"""
STEP 0 — DATA RECON
Inspect the raw police-violation CSV: shape, dtypes, nulls, uniques, date ranges,
value ranges, and the structure of the semi-structured columns (violation_type,
offence_code). Writes a machine-readable recon summary to outputs/tables/recon_summary.txt
and prints a human-readable report to stdout.

Run: .venv/bin/python src/00_recon.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = "/Users/arpitdhankani/Downloads/jan to may police violation_anonymized791b166(1).csv"
OUT = Path(__file__).resolve().parents[1] / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def section(title):
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}")


def main():
    section("LOADING")
    df = pd.read_csv(RAW, dtype=str, keep_default_na=False, na_values=["", "NULL", "null", "NaN"])
    print(f"Rows: {len(df):,}   Cols: {df.shape[1]}")
    print(f"Memory: {df.memory_usage(deep=True).sum()/1e6:.1f} MB")

    section("COLUMNS / DTYPES (as loaded = str)")
    for c in df.columns:
        print(f"  {c}")

    section("NULL % PER COLUMN")
    nullpct = (df.isna().mean() * 100).round(2).sort_values(ascending=False)
    print(nullpct.to_string())

    section("UNIQUE COUNTS PER COLUMN")
    nun = df.nunique(dropna=True).sort_values(ascending=False)
    print(nun.to_string())

    section("HEAD (first 5 rows, key cols)")
    keycols = ["id", "latitude", "longitude", "vehicle_type", "violation_type",
               "offence_code", "created_datetime", "closed_datetime", "police_station",
               "junction_name", "center_code", "validation_status"]
    print(df[keycols].head(5).to_string())

    # ---- Datetime parsing & ranges ----
    section("DATETIME RANGES")
    dt_cols = ["created_datetime", "closed_datetime", "modified_datetime",
               "action_taken_timestamp", "data_sent_to_scita_timestamp", "validation_timestamp"]
    dtinfo = {}
    for c in dt_cols:
        if c in df.columns:
            parsed = pd.to_datetime(df[c], errors="coerce", utc=True)
            nonnull = parsed.dropna()
            info = {
                "non_null": int(nonnull.shape[0]),
                "min": str(nonnull.min()) if len(nonnull) else None,
                "max": str(nonnull.max()) if len(nonnull) else None,
            }
            dtinfo[c] = info
            print(f"  {c:32s} non-null={info['non_null']:>8,}  min={info['min']}  max={info['max']}")

    # Monthly distribution of created_datetime
    section("created_datetime — MONTHLY COUNTS")
    cdt = pd.to_datetime(df["created_datetime"], errors="coerce", utc=True)
    monthly = cdt.dt.tz_convert("Asia/Kolkata").dt.to_period("M").value_counts().sort_index()
    print(monthly.to_string())

    # ---- Numeric ranges: lat/long ----
    section("COORDINATE RANGES (lat/long)")
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    print(f"  latitude : min={lat.min():.5f}  max={lat.max():.5f}  null={lat.isna().sum():,}  "
          f"zero={int((lat==0).sum()):,}")
    print(f"  longitude: min={lon.min():.5f}  max={lon.max():.5f}  null={lon.isna().sum():,}  "
          f"zero={int((lon==0).sum()):,}")
    # Bengaluru bbox sanity: lat ~12.7-13.2, lon ~77.3-77.8
    in_blr = ((lat.between(12.6, 13.3)) & (lon.between(77.3, 77.9)))
    print(f"  within Bengaluru bbox (12.6-13.3, 77.3-77.9): {in_blr.sum():,} "
          f"({in_blr.mean()*100:.2f}%)")

    # ---- Categorical value counts ----
    for col in ["vehicle_type", "validation_status", "data_sent_to_scita", "police_station"]:
        section(f"VALUE COUNTS — {col} (top 25)")
        vc = df[col].value_counts(dropna=False).head(25)
        print(vc.to_string())

    # ---- Parse violation_type (JSON array of strings) ----
    section("VIOLATION_TYPE — parsed (exploded) value counts")
    def parse_arr(x):
        if pd.isna(x):
            return []
        try:
            v = json.loads(x)
            return v if isinstance(v, list) else [v]
        except Exception:
            return [x]
    vt = df["violation_type"].apply(parse_arr)
    exploded = pd.Series([item for sub in vt for item in sub])
    print(f"  rows with >=1 violation: {(vt.apply(len)>0).sum():,}")
    print(f"  distinct violation labels: {exploded.nunique():,}")
    print(f"  multi-violation rows (>1 label): {(vt.apply(len)>1).sum():,}")
    print("\n  Top labels:")
    print(exploded.value_counts().head(30).to_string())

    section("OFFENCE_CODE — parsed value counts")
    oc = df["offence_code"].apply(parse_arr)
    oc_exp = pd.Series([str(i) for sub in oc for i in sub])
    print(oc_exp.value_counts().head(30).to_string())

    # ---- Junction coverage ----
    section("JUNCTION coverage")
    jn = df["junction_name"].fillna("MISSING")
    no_junction = (jn.str.strip().str.lower() == "no junction").sum()
    print(f"  'No Junction' rows: {no_junction:,} ({no_junction/len(df)*100:.1f}%)")
    print(f"  distinct named junctions (excl. 'No Junction'/missing): "
          f"{df.loc[~jn.str.strip().str.lower().isin(['no junction','missing']),'junction_name'].nunique():,}")
    print("\n  Top junctions:")
    print(df["junction_name"].value_counts().head(15).to_string())

    # ---- Duplicate / id checks ----
    section("ID & DUPLICATE CHECKS")
    print(f"  id unique: {df['id'].is_unique}  (distinct={df['id'].nunique():,} of {len(df):,})")
    dup_coords_time = df.duplicated(subset=["latitude", "longitude", "created_datetime",
                                            "vehicle_number"]).sum()
    print(f"  exact dup (lat,lon,created,vehicle): {dup_coords_time:,}")

    # ---- closed/resolution timing ----
    section("RESOLUTION TIMING (created -> closed)")
    closed = pd.to_datetime(df["closed_datetime"], errors="coerce", utc=True)
    created = cdt
    delta_h = (closed - created).dt.total_seconds() / 3600
    print(f"  rows with closed_datetime: {closed.notna().sum():,} "
          f"({closed.notna().mean()*100:.1f}%)")
    if closed.notna().sum() > 0:
        d = delta_h.dropna()
        d = d[(d >= 0) & (d < 24*365)]
        print(f"  resolution hours: median={d.median():.1f}  mean={d.mean():.1f}  "
              f"p90={d.quantile(0.9):.1f}")

    # ---- Save machine summary ----
    summary = {
        "rows": int(len(df)),
        "cols": list(df.columns),
        "null_pct": nullpct.round(2).to_dict(),
        "nunique": nun.to_dict(),
        "datetime_ranges": dtinfo,
        "monthly_counts": {str(k): int(v) for k, v in monthly.items()},
        "coord_in_blr_pct": float(in_blr.mean() * 100),
        "top_violation_labels": exploded.value_counts().head(30).to_dict(),
        "top_vehicle_types": df["vehicle_type"].value_counts().head(25).to_dict(),
        "top_police_stations": df["police_station"].value_counts().head(40).to_dict(),
        "no_junction_pct": float(no_junction / len(df) * 100),
        "validation_status": df["validation_status"].value_counts(dropna=False).to_dict(),
    }
    with open(OUT / "recon_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[saved] {OUT/'recon_summary.json'}")


if __name__ == "__main__":
    main()
