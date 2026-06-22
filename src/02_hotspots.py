"""
02 — HOTSPOT DETECTION (Objective 1)
  (a) DBSCAN (haversine) micro-hotspots — tight clusters of illegal parking with centroids.
  (b) ~250 m grid aggregation — fine hotspots independent of admin boundaries.
  (c) Zone (police_station) & junction rollups — operational units.
  (d) Temporal patterns — day-of-week, month (trustworthy) + hour-of-day (operational, caveated).
Saves tables to outputs/tables/ and figures to outputs/figures/.

Run: .venv/bin/python src/02_hotspots.py
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import config as C

sns.set_theme(style="whitegrid", context="talk")
np.random.seed(C.SEED)
PRIMARY = "#1f4e79"; ACCENT = "#c0392b"; ORANGE = "#e67e22"


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {name}")


def main():
    df = pd.read_parquet(C.CLEAN_PARQUET)
    conf = df[df["is_confirmed"]].copy()
    print(f"confirmed events: {len(conf):,}")

    # ============================================================ (a) DBSCAN micro-hotspots
    print("\n[DBSCAN] clustering confirmed coordinates (haversine)…")
    coords = np.radians(conf[["latitude", "longitude"]].values)
    eps = C.DBSCAN_EPS_M / C.EARTH_RADIUS_M
    db = DBSCAN(eps=eps, min_samples=C.DBSCAN_MIN_SAMPLES, metric="haversine",
                algorithm="ball_tree", n_jobs=-1).fit(coords)
    conf["cluster"] = db.labels_
    n_clusters = int((conf["cluster"] >= 0).sum() and conf.loc[conf.cluster >= 0, "cluster"].nunique())
    n_noise = int((conf["cluster"] == -1).sum())
    print(f"  clusters: {n_clusters:,}   clustered pts: {len(conf)-n_noise:,}   noise: {n_noise:,}")

    cl = conf[conf["cluster"] >= 0].groupby("cluster").agg(
        n_events=("id", "size"),
        weighted=("congestion_weight", "sum"),
        high_sev=("is_high_severity", "sum"),
        lat=("latitude", "mean"),
        lon=("longitude", "mean"),
        active_days=("date", "nunique"),
        zone=("police_station", lambda s: s.value_counts().index[0] if len(s) else "-"),
        junction=("junction_clean", lambda s: s.value_counts().index[0] if len(s) else "-"),
        top_violation=("primary_violation", lambda s: s.value_counts().index[0] if len(s) else "-"),
    ).reset_index()
    cl["events_per_day"] = (cl["n_events"] / cl["active_days"]).round(2)
    cl = cl.sort_values("weighted", ascending=False).reset_index(drop=True)
    cl.insert(0, "rank", cl.index + 1)
    cl.to_csv(C.TBL_DIR / "dbscan_hotspots.csv", index=False)
    print(f"  [tbl] dbscan_hotspots.csv  (top cluster: {cl.iloc[0]['n_events']:,} events @ "
          f"{cl.iloc[0]['zone']})")

    # ============================================================ (b) grid hotspots
    grid = conf.groupby(["grid_cell", "grid_lat", "grid_lon"]).agg(
        n_events=("id", "size"),
        weighted=("congestion_weight", "sum"),
        high_sev=("is_high_severity", "sum"),
        active_days=("date", "nunique"),
        zone=("police_station", lambda s: s.value_counts().index[0] if len(s) else "-"),
    ).reset_index().sort_values("weighted", ascending=False)
    grid["cum_share"] = (grid["n_events"].cumsum() / grid["n_events"].sum()).round(4)
    grid.to_csv(C.TBL_DIR / "grid_hotspots.csv", index=False)
    n_cells = len(grid)
    # spatial concentration: share of events in top 1% of cells
    top1pct = max(1, int(round(n_cells * 0.01)))
    conc_top1 = grid.head(top1pct)["n_events"].sum() / grid["n_events"].sum()
    print(f"  [tbl] grid_hotspots.csv  cells={n_cells:,}  "
          f"top-1% cells hold {conc_top1*100:.1f}% of events")

    # ============================================================ (c) zone & junction rollups
    def rollup(g):
        r = g.agg(
            n_events=("id", "size"),
            weighted=("congestion_weight", "sum"),
            high_sev_events=("is_high_severity", "sum"),
            main_road_events=("is_main_road", "sum"),
            junction_events=("has_junction", "sum"),
            active_days=("date", "nunique"),
            active_cells=("grid_cell", "nunique"),
            top_violation=("primary_violation", lambda s: s.value_counts().index[0] if len(s) else "-"),
            top_vehicle=("vehicle_type", lambda s: s.value_counts().index[0] if len(s) else "-"),
            peak_dow=("dow_name", lambda s: s.value_counts().index[0] if len(s) else "-"),
        )
        r["high_sev_share"] = (r["high_sev_events"] / r["n_events"]).round(3)
        r["events_per_active_day"] = (r["n_events"] / r["active_days"]).round(1)
        r["events_per_cell"] = (r["n_events"] / r["active_cells"]).round(1)
        return r.sort_values("weighted", ascending=False)

    zone = rollup(conf.groupby("police_station")).reset_index()
    zone.to_csv(C.TBL_DIR / "zone_stats.csv", index=False)
    print(f"  [tbl] zone_stats.csv  ({len(zone)} zones; top: {zone.iloc[0]['police_station']})")

    jdf = conf[conf["has_junction"]]
    junc = rollup(jdf.groupby("junction_clean")).reset_index()
    junc.to_csv(C.TBL_DIR / "junction_stats.csv", index=False)
    print(f"  [tbl] junction_stats.csv  ({len(junc)} junctions)")

    # ============================================================ (d) temporal patterns
    hour = conf["hour"].value_counts(normalize=True).sort_index() * 100
    dow = conf.groupby(["dow", "dow_name"]).size().reset_index(name="n").sort_values("dow")
    monthly = conf.groupby("month").size()
    pd.DataFrame({"hour": hour.index, "pct": hour.values}).to_csv(C.TBL_DIR / "temporal_hour.csv", index=False)
    dow.to_csv(C.TBL_DIR / "temporal_dow.csv", index=False)
    monthly.to_csv(C.TBL_DIR / "temporal_month.csv")

    # ---------- FIGURES ----------
    print("\n[figures]")
    # 1. spatial density heatmap (PNG snapshot)
    fig, ax = plt.subplots(figsize=(9, 9))
    hb = ax.hexbin(conf["longitude"], conf["latitude"], gridsize=120, mincnt=1,
                   bins="log", cmap="inferno")
    ax.set_title("Illegal-Parking Violation Density — Bengaluru\n(confirmed events, log scale)",
                 fontsize=15)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_aspect("equal")
    cb = fig.colorbar(hb, ax=ax, shrink=0.7); cb.set_label("events (log10)")
    savefig(fig, "fig_violation_heatmap.png")

    # 2. hour-of-day (operational, caveated)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(hour.index, hour.values, color=PRIMARY)
    ax.axvspan(14.5, 23.5, color=ACCENT, alpha=0.10)
    ax.text(19, hour.max()*0.7, "EVENING VISIBILITY GAP\n(~1.4% of records)",
            ha="center", color=ACCENT, fontsize=11, weight="bold")
    ax.set_title("When violations are RECORDED (IST) — enforcement window, not true demand")
    ax.set_xlabel("Hour of day (IST)"); ax.set_ylabel("% of confirmed records")
    ax.set_xticks(range(0, 24))
    savefig(fig, "fig_hour_of_day.png")

    # 3. day-of-week
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [ACCENT if d in (5, 6) else PRIMARY for d in dow["dow"]]
    ax.bar(dow["dow_name"], dow["n"], color=colors)
    ax.set_title("Violations by day of week (weekend in red)")
    ax.set_ylabel("confirmed events"); plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    savefig(fig, "fig_day_of_week.png")

    # 4. monthly trend
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(monthly)), monthly.values, "o-", color=PRIMARY, lw=2.5, ms=9)
    ax.set_xticks(range(len(monthly))); ax.set_xticklabels(monthly.index, rotation=30, ha="right")
    ax.set_title("Monthly confirmed violations (Apr-24 partial)")
    ax.set_ylabel("events")
    for i, v in enumerate(monthly.values):
        ax.annotate(f"{v:,}", (i, v), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10)
    savefig(fig, "fig_monthly_trend.png")

    # 5. top zones
    topz = zone.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(topz["police_station"], topz["n_events"], color=PRIMARY)
    ax.set_title("Top 15 enforcement zones by confirmed violations")
    ax.set_xlabel("confirmed events")
    savefig(fig, "fig_top_zones.png")

    # 6. top junctions
    topj = junc.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(topj["junction_clean"].str.replace(r"BTP\d+ - ", "", regex=True), topj["n_events"],
            color=ORANGE)
    ax.set_title("Top 15 junctions by confirmed violations")
    ax.set_xlabel("confirmed events")
    savefig(fig, "fig_top_junctions.png")

    # 7. violation & vehicle mix
    vt = conf["primary_violation"].value_counts().head(10).iloc[::-1]
    vh = conf["vehicle_type"].value_counts().head(10).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].barh(vt.index, vt.values, color=PRIMARY); axes[0].set_title("Primary violation (top 10)")
    axes[1].barh(vh.index, vh.values, color=ORANGE); axes[1].set_title("Vehicle type (top 10)")
    savefig(fig, "fig_violation_vehicle_mix.png")

    # save headline temporal numbers
    summary = {
        "n_confirmed": int(len(conf)),
        "dbscan_clusters": n_clusters,
        "dbscan_noise_pct": round(n_noise / len(conf) * 100, 1),
        "grid_cells": int(n_cells),
        "grid_top1pct_concentration": round(conc_top1 * 100, 1),
        "evening_gap_pct": round(conf["hour"].between(15, 23).mean() * 100, 2),
        "peak_record_hours": [int(h) for h in hour.sort_values(ascending=False).head(4).index],
        "weekend_share": round(conf["is_weekend"].mean() * 100, 1),
    }
    pd.Series(summary).to_json(C.TBL_DIR / "hotspot_summary.json")
    print(f"\n[saved] hotspot_summary.json  -> {summary}")


if __name__ == "__main__":
    main()
