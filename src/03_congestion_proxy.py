"""
03 — CONGESTION IMPACT PROXY (Objective 2)
There is NO traffic-flow data (speed/volume/delay). We therefore MODEL a congestion-impact
proxy and label it as such. Every event gets transparent "congestion-impact units":

    impact = congestion_weight * (1 + J_BOOST*at_junction + M_BOOST*on_main_road)

Rationale (all data-derived axes): a heavier offence (weight 3 vs 2), one that sits AT AN
INTERSECTION, or one ON A MAIN ROAD removes more live carriageway capacity. We then:
  - roll impact up to zone / junction / grid (Congestion Impact Load, CIL),
  - report EFFECT SIZES (junction vs non-junction severity; volume<->intersection link),
  - measure spatial concentration (Gini / Lorenz),
  - run a SENSITIVITY check: re-rank zones under alternative weightings (Spearman rho).

Run: .venv/bin/python src/03_congestion_proxy.py
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import config as C

sns.set_theme(style="whitegrid", context="talk")
PRIMARY = "#1f4e79"; ACCENT = "#c0392b"; ORANGE = "#e67e22"; GREEN = "#27ae60"


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    if n == 0 or x.sum() == 0:
        return np.nan
    return (2 * np.sum((np.arange(1, n + 1)) * x) / (n * x.sum())) - (n + 1) / n


def savefig(fig, name):
    fig.tight_layout(); fig.savefig(C.FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"  [fig] {name}")


def main():
    df = pd.read_parquet(C.CLEAN_PARQUET)
    conf = df[df["is_confirmed"]].copy()

    # ---- per-event congestion-impact units (the proxy core) ----
    conf["impact"] = conf["congestion_weight"] * (
        1
        + C.IMPACT_JUNCTION_BOOST * conf["has_junction"].astype(int)
        + C.IMPACT_MAINROAD_BOOST * conf["is_main_road"].astype(int)
    )
    print(f"confirmed events: {len(conf):,}   total impact units: {conf['impact'].sum():,.0f}")

    # ================================================== EFFECT SIZES (honest, data-derived)
    print("\n[EFFECT SIZES]")
    # 1. high-severity rate at junction vs not
    hj = conf.loc[conf.has_junction, "is_high_severity"].mean()
    hn = conf.loc[~conf.has_junction, "is_high_severity"].mean()
    rr = hj / hn if hn else np.nan
    # main-road rate
    mj = conf.loc[conf.has_junction, "is_main_road"].mean()
    mn = conf.loc[~conf.has_junction, "is_main_road"].mean()
    print(f"  high-severity rate: junction={hj*100:.1f}% vs non-junction={hn*100:.1f}%  (RR={rr:.2f}x)")
    print(f"  main-road rate    : junction={mj*100:.1f}% vs non-junction={mn*100:.1f}%  (RR={mj/mn:.2f}x)")

    # 2. zone-level: does violation volume track intersection exposure?
    zg = conf.groupby("police_station").agg(
        volume=("id", "size"),
        impact=("impact", "sum"),
        junction_events=("has_junction", "sum"),
        main_road_events=("is_main_road", "sum"),
        high_sev_events=("is_high_severity", "sum"),
        active_days=("date", "nunique"),
        active_cells=("grid_cell", "nunique"),
    ).reset_index()
    zg["junction_share"] = zg["junction_events"] / zg["volume"]
    zg["impact_per_event"] = zg["impact"] / zg["volume"]
    zg["density"] = zg["volume"] / zg["active_cells"]
    rho_vol_junc, p1 = stats.spearmanr(zg["volume"], zg["junction_share"])
    rho_vol_imp, p2 = stats.spearmanr(zg["volume"], zg["impact"])
    print(f"  Spearman(zone volume, junction-share) = {rho_vol_junc:.2f} (p={p1:.3f})")
    print(f"  Spearman(zone volume, impact load)    = {rho_vol_imp:.2f} (p={p2:.3f})")

    # ================================================== CONGESTION IMPACT LOAD by unit
    zone_cil = zg.sort_values("impact", ascending=False).reset_index(drop=True)
    zone_cil.to_csv(C.TBL_DIR / "zone_congestion_impact.csv", index=False)

    jcil = (conf[conf.has_junction].groupby("junction_clean")
            .agg(volume=("id", "size"), impact=("impact", "sum"),
                 high_sev_events=("is_high_severity", "sum"),
                 main_road_events=("is_main_road", "sum"),
                 active_days=("date", "nunique"))
            .sort_values("impact", ascending=False).reset_index())
    jcil.to_csv(C.TBL_DIR / "junction_congestion_impact.csv", index=False)

    grid_cil = (conf.groupby(["grid_cell", "grid_lat", "grid_lon"])
                .agg(volume=("id", "size"), impact=("impact", "sum"))
                .sort_values("impact", ascending=False).reset_index())
    grid_cil.to_csv(C.TBL_DIR / "grid_congestion_impact.csv", index=False)

    # ================================================== CONCENTRATION (Gini / Lorenz)
    g_zone = gini(zone_cil["impact"])
    g_grid = gini(grid_cil["impact"])
    print(f"\n[CONCENTRATION]  Gini(impact across zones)={g_zone:.3f}   "
          f"Gini(impact across grid cells)={g_grid:.3f}")

    # ================================================== SENSITIVITY (rank robustness)
    print("\n[SENSITIVITY] zone impact ranking under alternative weight schemes:")
    base_rank = zone_cil.set_index("police_station")["impact"].rank(ascending=False)
    # scheme A: ignore boosts (severity-weighted volume only)
    altA = conf.groupby("police_station")["congestion_weight"].sum()
    # scheme B: flat count (every event weight 1, no boosts)
    altB = conf.groupby("police_station").size()
    # scheme C: bigger boosts (1.0 each)
    impC = conf["congestion_weight"] * (1 + 1.0*conf.has_junction.astype(int) + 1.0*conf.is_main_road.astype(int))
    altC = impC.groupby(conf["police_station"]).sum()
    rows = []
    for nm, s in [("severity-only", altA), ("flat-count", altB), ("double-boost", altC)]:
        rho, _ = stats.spearmanr(base_rank, s.rank(ascending=False).reindex(base_rank.index))
        rows.append((nm, round(rho, 3)))
        print(f"  Spearman(base, {nm:13s}) = {rho:.3f}")
    pd.DataFrame(rows, columns=["scheme", "spearman_vs_base"]).to_csv(
        C.TBL_DIR / "congestion_sensitivity.csv", index=False)

    # ---------- FIGURES ----------
    print("\n[figures]")
    # congestion-impact heatmap (weighted hexbin)
    fig, ax = plt.subplots(figsize=(9, 9))
    hb = ax.hexbin(conf["longitude"], conf["latitude"], C=conf["impact"],
                   reduce_C_function=np.sum, gridsize=120, mincnt=1, bins="log", cmap="magma")
    ax.set_title("Congestion-Impact Load (proxy) — Bengaluru\nseverity x intersection x main-road, log scale",
                 fontsize=14)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_aspect("equal")
    cb = fig.colorbar(hb, ax=ax, shrink=0.7); cb.set_label("impact units (log10)")
    savefig(fig, "fig_congestion_impact_heatmap.png")

    # Lorenz curve (grid concentration)
    x = np.sort(grid_cil["impact"].values)
    cum = np.cumsum(x) / x.sum(); p = np.arange(1, len(x)+1) / len(x)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot(p, cum, color=ACCENT, lw=3, label=f"Lorenz (Gini={g_grid:.2f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect equality")
    ax.fill_between(p, cum, p, color=ACCENT, alpha=0.10)
    ax.set_title("Spatial concentration of congestion impact\n(~250 m grid cells)")
    ax.set_xlabel("cumulative share of grid cells"); ax.set_ylabel("cumulative share of impact")
    ax.legend(loc="upper left")
    savefig(fig, "fig_lorenz_concentration.png")

    # effect size: junction vs non-junction (HONEST finding — severity is higher OFF junctions)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    cats = ["High-severity\n(weight-3) violation", "Main-road\nparking"]
    jvals = [hj*100, mj*100]; nvals = [hn*100, mn*100]
    x = np.arange(2); w = 0.38
    ax.bar(x - w/2, nvals, w, label="On open carriageway (no junction)", color=ACCENT)
    ax.bar(x + w/2, jvals, w, label="At a named junction", color=PRIMARY)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("% of events"); ax.set_ylim(0, 16.5)
    ax.set_title("Finding: heavy-impact parking sits on OPEN carriageways,\nnot at named junctions "
                 f"(RR {1/rr:.1f}x)", fontsize=13)
    for i, (b, a) in enumerate(zip(nvals, jvals)):
        ax.text(i - w/2, b + 0.3, f"{b:.1f}%", ha="center", fontsize=11)
        ax.text(i + w/2, a + 0.3, f"{a:.1f}%", ha="center", fontsize=11)
    ax.legend(loc="upper center", ncol=2, framealpha=0.95, fontsize=11)
    savefig(fig, "fig_effect_junction_severity.png")

    # zone volume vs impact scatter
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(zone_cil["volume"], zone_cil["impact"], s=zone_cil["junction_share"]*400+30,
                    c=zone_cil["junction_share"], cmap="viridis", alpha=0.8, edgecolor="k", lw=0.5)
    for _, r in zone_cil.head(6).iterrows():
        ax.annotate(r["police_station"], (r["volume"], r["impact"]), fontsize=9,
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("violation volume"); ax.set_ylabel("congestion-impact load (units)")
    ax.set_title("Zones: volume vs congestion impact\n(size/colour = intersection exposure)")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("junction share")
    savefig(fig, "fig_zone_volume_vs_impact.png")

    summary = {
        "total_impact_units": round(float(conf["impact"].sum()), 0),
        "hi_sev_rate_junction": round(hj*100, 1),
        "hi_sev_rate_nonjunction": round(hn*100, 1),
        "hi_sev_relative_risk": round(rr, 2),
        "mainroad_rate_junction": round(mj*100, 1),
        "mainroad_rate_nonjunction": round(mn*100, 1),
        "spearman_volume_junctionshare": round(rho_vol_junc, 2),
        "gini_impact_zones": round(g_zone, 3),
        "gini_impact_grid": round(g_grid, 3),
        "sensitivity_min_spearman": round(min(r[1] for r in rows), 3),
    }
    pd.Series(summary).to_json(C.TBL_DIR / "congestion_summary.json")
    print(f"\n[saved] congestion_summary.json -> {summary}")


if __name__ == "__main__":
    main()
