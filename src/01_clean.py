"""
01 — CLEAN & FEATURE-BUILD
Load raw CSV, parse semi-structured columns, convert time to IST, derive per-event
congestion severity, flag validity/duplicates, and snap each event to a ~250 m grid cell.
Writes data/clean_violations.parquet (all rows + flags; downstream filters on is_confirmed).

Run: .venv/bin/python src/01_clean.py
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import config as C


def parse_arr(x):
    if pd.isna(x):
        return []
    try:
        v = json.loads(x)
        return [str(i).strip() for i in v] if isinstance(v, list) else [str(v).strip()]
    except Exception:
        return [str(x).strip()]


def main():
    print("Loading raw CSV …")
    df = pd.read_csv(C.RAW_CSV, dtype=str, keep_default_na=False,
                     na_values=["", "NULL", "null", "NaN"])
    n0 = len(df)
    print(f"  raw rows: {n0:,}")

    # --- coordinates ---
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    in_box = (df["latitude"].between(C.BBOX["lat_min"], C.BBOX["lat_max"]) &
              df["longitude"].between(C.BBOX["lon_min"], C.BBOX["lon_max"]))
    df = df[in_box].copy()
    print(f"  in Bengaluru bbox: {len(df):,} (dropped {n0-len(df):,})")

    # --- time (UTC -> IST) ---
    ts = pd.to_datetime(df["created_datetime"], errors="coerce", utc=True)
    df = df[ts.notna()].copy()
    ts = ts[ts.notna()]
    ist = ts.dt.tz_convert(C.IST)
    df["ts_ist"] = ist.dt.tz_localize(None)            # naive IST for parquet friendliness
    df["date"] = ist.dt.date.astype("datetime64[ns]")
    df["hour"] = ist.dt.hour.astype("int16")
    df["dow"] = ist.dt.dayofweek.astype("int16")       # Mon=0
    df["dow_name"] = ist.dt.day_name()
    df["month"] = ist.dt.to_period("M").astype(str)
    df["is_weekend"] = df["dow"].isin([5, 6])
    df["iso_week"] = ist.dt.tz_localize(None).dt.to_period("W-SUN").astype(str)
    print(f"  with valid timestamp: {len(df):,}")

    # --- violation labels & congestion severity ---
    vt = df["violation_type"].apply(parse_arr)
    df["violation_list"] = vt
    df["n_violations"] = vt.apply(len).astype("int16")
    df["violation_labels"] = vt.apply(lambda L: "; ".join(L))

    def event_weight(labels):
        if not labels:
            return C.DEFAULT_WEIGHT
        return max(C.CONGESTION_WEIGHTS.get(l, C.DEFAULT_WEIGHT) for l in labels)

    def primary_label(labels):
        if not labels:
            return "UNKNOWN"
        return max(labels, key=lambda l: C.CONGESTION_WEIGHTS.get(l, C.DEFAULT_WEIGHT))

    df["congestion_weight"] = vt.apply(event_weight).astype("int16")
    df["primary_violation"] = vt.apply(primary_label)
    df["is_parking"] = vt.apply(lambda L: any(l not in C.NON_PARKING_LABELS for l in L) if L else True)
    df["is_high_severity"] = df["congestion_weight"] >= 3

    # --- junction / zone ---
    j = df["junction_name"].fillna("No Junction").str.strip()
    df["has_junction"] = ~j.str.lower().isin(["no junction", ""])
    df["junction_clean"] = np.where(df["has_junction"], j, "No Junction")
    df["police_station"] = df["police_station"].fillna("UNKNOWN").str.strip()

    # main-road flag (independent, data-derived congestion axis)
    df["is_main_road"] = vt.apply(lambda L: "PARKING IN A MAIN ROAD" in L)

    # --- validation / duplicates ---
    vs = df["validation_status"].fillna("unvalidated").str.lower()
    df["validation_status_clean"] = vs
    df["is_rejected"] = vs.eq("rejected")
    df["is_approved"] = vs.eq("approved")
    df["is_dup_status"] = vs.eq("duplicate")

    df["is_exact_dup"] = df.duplicated(
        subset=["vehicle_number", "latitude", "longitude", "created_datetime"], keep="first")

    # confirmed analysis set: drop rejected/duplicate-status and exact dupes
    df["is_confirmed"] = (~vs.isin(C.DROP_VALIDATION)) & (~df["is_exact_dup"])

    # --- spatial grid ---
    glat, glon = C.grid_id(df["latitude"].values, df["longitude"].values)
    df["grid_lat"] = glat
    df["grid_lon"] = glon
    df["grid_cell"] = (pd.Series(glat, index=df.index).round(5).astype(str) + "," +
                       pd.Series(glon, index=df.index).round(5).astype(str))

    # --- persist (drop the python-list col; keep string version) ---
    # --- persist (drop the python-list col; keep string version) ---
    keep = ["id", "latitude", "longitude", "location", "vehicle_type",
            "primary_violation", "violation_labels", "n_violations", "offence_code",
            "congestion_weight", "is_high_severity", "is_parking", "is_main_road",
            "police_station", "center_code", "junction_clean", "has_junction",
            "ts_ist", "date", "hour", "dow", "dow_name", "month", "iso_week", "is_weekend",
            "validation_status_clean", "is_rejected", "is_approved",
            "is_exact_dup", "is_confirmed", "grid_lat", "grid_lon", "grid_cell"]
    out = df[keep].copy()
    
   # --- NEW: RUN COMPUTER VISION PIPELINE AND MERGE ---
    print("\n==> Launching AI Computer Vision Pipeline...")
    import importlib
    cv_pipeline = importlib.import_module("07_cv_pipeline")
    cv_violations_df = cv_pipeline.process_parking_feed()
    
    # FIX: Align columns so the downstream scripts don't crash on NaNs
    cv_violations_df["is_confirmed"] = True
    cv_violations_df["grid_lat"] = 12.9716
    cv_violations_df["grid_lon"] = 77.5946
    cv_violations_df["grid_cell"] = "12.9716,77.5946"
    cv_violations_df["has_junction"] = False
    cv_violations_df["junction_clean"] = "No Junction"
    cv_violations_df["congestion_weight"] = 3
    cv_violations_df["is_main_road"] = True 
    cv_violations_df["is_parking"] = True
    cv_violations_df["is_high_severity"] = True
    cv_violations_df["ts_ist"] = cv_violations_df["created_datetime"]
    cv_violations_df["date"] = cv_violations_df["created_datetime"].dt.date.astype("datetime64[ns]")
    cv_violations_df["hour"] = cv_violations_df["created_datetime"].dt.hour.astype("int16")
    cv_violations_df["dow"] = cv_violations_df["created_datetime"].dt.dayofweek.astype("int16")
    cv_violations_df["dow_name"] = cv_violations_df["created_datetime"].dt.day_name()
    cv_violations_df["month"] = cv_violations_df["created_datetime"].dt.to_period("M").astype(str)
    cv_violations_df["is_weekend"] = cv_violations_df["dow"].isin([5, 6])
    cv_violations_df["iso_week"] = cv_violations_df["created_datetime"].dt.to_period("W-SUN").astype(str)
    # Merge the live CV data with the historical CSV data
    out = pd.concat([out, cv_violations_df], ignore_index=True)
    
    # Force the column to be strictly True/False to prevent the ValueError
    out["is_confirmed"] = out["is_confirmed"].astype(bool)
    # ---------------------------------------------------

    out.to_parquet(C.CLEAN_PARQUET, index=False)

    # --- report ---
    print("\n--- CLEAN SUMMARY ---")
    print(f"  rows persisted        : {len(out):,}")
    print(f"  confirmed set         : {out['is_confirmed'].sum():,} "
          f"({out['is_confirmed'].mean()*100:.1f}%)")
    print(f"  rejected (excluded)   : {out['is_rejected'].sum():,}")
    print(f"  exact dups (excluded) : {out['is_exact_dup'].sum():,}")
    print(f"  at named junction     : {out['has_junction'].mean()*100:.1f}%")
    print(f"  high-severity events  : {out['is_high_severity'].mean()*100:.1f}%")
    print(f"  congestion_weight mix :\n{out['congestion_weight'].value_counts().sort_index().to_string()}")
    print(f"\n[saved] {C.CLEAN_PARQUET}")


if __name__ == "__main__":
    main()
