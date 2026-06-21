"""
04 — ENFORCEMENT PRIORITIZATION (Objective 3)
Composite Enforcement Priority Index (EPI) per zone and per junction:

    EPI = 100 * Σ w_k * minmax(component_k)
    components = volume (0.30) + congestion-impact (0.35) + recurrence (0.20) + intensity (0.15)

Outputs a ranked, ACTIONABLE table: top-N zones, what drives the score, peak day/window,
and the cumulative citywide congestion-impact each tier addresses.

Run: .venv/bin/python src/04_prioritization.py
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import config as C

sns.set_theme(style="whitegrid", context="talk")
PRIMARY = "#1f4e79"; ACCENT = "#c0392b"; ORANGE = "#e67e22"; GREEN = "#27ae60"


def minmax(s):
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else s * 0.0


def savefig(fig, name):
    fig.tight_layout(); fig.savefig(C.FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"  [fig] {name}")


def add_impact(df):
    df = df.copy()
    df["impact"] = df["congestion_weight"] * (
        1 + C.IMPACT_JUNCTION_BOOST * df["has_junction"].astype(int)
        + C.IMPACT_MAINROAD_BOOST * df["is_main_road"].astype(int))
    return df


def build_index(conf, key):
    g = conf.groupby(key).agg(
        volume=("id", "size"),
        impact=("impact", "sum"),
        active_days=("date", "nunique"),
        active_cells=("grid_cell", "nunique"),
        high_sev_events=("is_high_severity", "sum"),
        junction_events=("has_junction", "sum"),
        top_violation=("primary_violation", lambda s: s.mode().iat[0]),
        top_vehicle=("vehicle_type", lambda s: s.mode().iat[0]),
        peak_dow=("dow_name", lambda s: s.mode().iat[0]),
        peak_hour=("hour", lambda s: int(s.mode().iat[0])),
    ).reset_index()
    g["intensity"] = g["volume"] / g["active_cells"]
    # components
    g["c_volume"] = minmax(g["volume"])
    g["c_impact"] = minmax(g["impact"])
    g["c_recur"] = minmax(g["active_days"])
    g["c_intensity"] = minmax(g["intensity"])
    w = C.PRIORITY_WEIGHTS
    g["priority_score"] = 100 * (
        w["violation_volume"] * g["c_volume"]
        + w["congestion_impact"] * g["c_impact"]
        + w["recurrence"] * g["c_recur"]
        + w["spatial_intensity"] * g["c_intensity"])
    g = g.sort_values("priority_score", ascending=False).reset_index(drop=True)
    g.insert(0, "rank", g.index + 1)
    # citywide coverage
    g["impact_share"] = g["impact"] / g["impact"].sum()
    g["cum_impact_share"] = g["impact_share"].cumsum()
    g["volume_share"] = g["volume"] / g["volume"].sum()
    g["cum_volume_share"] = g["volume_share"].cumsum()
    return g


def main():
    conf = add_impact(pd.read_parquet(C.CLEAN_PARQUET).query("is_confirmed"))
    print(f"confirmed events: {len(conf):,}")

    # ---------- ZONE index ----------
    zone = build_index(conf, "police_station")
    # recommended action string
    def rec(r):
        ev = "extend to evening (current records stop ~14:00 IST)" \
            if r["priority_score"] >= zone["priority_score"].quantile(0.8) else "peak-window sweep"
        return f"Patrol {r['peak_dow']}; {ev}; target {r['top_violation'].lower()} / {r['top_vehicle'].lower()}"
    zone["recommended_action"] = zone.apply(rec, axis=1)
    zone.to_csv(C.TBL_DIR / "zone_priority.csv", index=False)

    cols = ["rank", "police_station", "priority_score", "volume", "impact", "active_days",
            "high_sev_events", "junction_events", "intensity", "cum_impact_share",
            "peak_dow", "top_violation", "top_vehicle", "recommended_action"]
    top = zone[cols].head(C.TOP_N_ZONES).copy()
    top["priority_score"] = top["priority_score"].round(1)
    top["intensity"] = top["intensity"].round(1)
    top["cum_impact_share"] = (top["cum_impact_share"] * 100).round(1)
    top.to_csv(C.TBL_DIR / "top_enforcement_zones.csv", index=False)

    print("\n=== TOP 10 ENFORCEMENT ZONES ===")
    print(top.head(10)[["rank", "police_station", "priority_score", "volume",
                        "impact", "active_days", "cum_impact_share", "peak_dow"]].to_string(index=False))

    cov5 = zone.head(5)["impact_share"].sum() * 100
    cov10 = zone.head(10)["impact_share"].sum() * 100
    cov15 = zone.head(15)["impact_share"].sum() * 100
    print(f"\nCoverage: top-5 zones = {cov5:.1f}% of citywide congestion impact | "
          f"top-10 = {cov10:.1f}% | top-15 = {cov15:.1f}%  (of {zone.shape[0]} zones)")

    # ---------- JUNCTION index ----------
    junc = build_index(conf[conf.has_junction], "junction_clean")
    junc.to_csv(C.TBL_DIR / "junction_priority.csv", index=False)
    jtop = junc.head(C.TOP_N_JUNCTIONS)
    jcov10 = junc.head(10)["impact_share"].sum() * 100
    print(f"\nTop junction: {jtop.iloc[0]['junction_clean']} (score {jtop.iloc[0]['priority_score']:.1f}); "
          f"top-10 junctions = {jcov10:.1f}% of junction impact")

    # ---------- FIGURES ----------
    print("\n[figures]")
    # priority ranking bar
    tz = zone.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(tz["police_station"], tz["priority_score"],
                   color=plt.cm.Reds(0.4 + 0.5 * minmax(tz["priority_score"])))
    ax.set_title("Enforcement Priority Index — Top 15 zones")
    ax.set_xlabel("priority score (0–100)")
    for b, v in zip(bars, tz["priority_score"]):
        ax.text(v + 0.5, b.get_y() + b.get_height()/2, f"{v:.0f}", va="center", fontsize=10)
    savefig(fig, "fig_priority_ranking.png")

    # component stack for top 10
    t10 = zone.head(10).iloc[::-1]
    w = C.PRIORITY_WEIGHTS
    comp = pd.DataFrame({
        "Volume": 100*w["violation_volume"]*t10["c_volume"].values,
        "Congestion impact": 100*w["congestion_impact"]*t10["c_impact"].values,
        "Recurrence": 100*w["recurrence"]*t10["c_recur"].values,
        "Spatial intensity": 100*w["spatial_intensity"]*t10["c_intensity"].values,
    }, index=t10["police_station"].values)
    fig, ax = plt.subplots(figsize=(11, 7))
    comp.plot(kind="barh", stacked=True, ax=ax,
              color=[PRIMARY, ACCENT, ORANGE, GREEN])
    ax.set_title("What drives priority — component contribution (top 10 zones)")
    ax.set_xlabel("priority score contribution"); ax.legend(loc="lower right", fontsize=11)
    savefig(fig, "fig_priority_components.png")

    # cumulative coverage curve
    fig, ax = plt.subplots(figsize=(9, 6))
    n = np.arange(1, len(zone)+1)
    ax.plot(n, zone["cum_impact_share"]*100, "o-", color=ACCENT, ms=4, label="congestion impact")
    ax.plot(n, zone["cum_volume_share"]*100, "s-", color=PRIMARY, ms=4, label="violation volume")
    for k in (5, 10, 15):
        ax.axvline(k, color="grey", ls=":", lw=1)
        ax.text(k, 20, f"top {k}", rotation=90, va="bottom", fontsize=9, color="grey")
    ax.set_title("Enforcement coverage: targeting top-N zones")
    ax.set_xlabel("number of top zones patrolled"); ax.set_ylabel("% of citywide total covered")
    ax.legend(loc="lower right")
    savefig(fig, "fig_coverage_curve.png")

    summary = {
        "n_zones": int(len(zone)),
        "n_junctions": int(len(junc)),
        "top_zone": zone.iloc[0]["police_station"],
        "top_zone_score": round(float(zone.iloc[0]["priority_score"]), 1),
        "top5_impact_coverage_pct": round(cov5, 1),
        "top10_impact_coverage_pct": round(cov10, 1),
        "top15_impact_coverage_pct": round(cov15, 1),
        "top_junction": junc.iloc[0]["junction_clean"],
        "top10_junction_coverage_pct": round(jcov10, 1),
        "priority_weights": C.PRIORITY_WEIGHTS,
    }
    pd.Series(summary).to_json(C.TBL_DIR / "priority_summary.json")
    print(f"\n[saved] priority_summary.json -> top5 covers {cov5:.1f}% of impact")


if __name__ == "__main__":
    main()
