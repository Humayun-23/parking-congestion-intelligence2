"""
05 — SHORT FORECAST (Objective 4)
Predict NEXT-WEEK violation load per zone (a "hotspot-risk" signal for patrol planning).
Honest, explainable, leakage-safe:
  - weekly zone panel (zone x week, zero-filled); first & last partial weeks dropped.
  - features use ONLY past info: lag-1/2/3/4, shifted 4-week rolling mean, week trend, month,
    zone identity (categorical) — no target leakage.
  - time-based split: last 4 weeks = test.
  - model: HistGradientBoostingRegressor (Poisson) vs Ridge vs two naive baselines
    (seasonal-naive = last week; 4-week moving average).
  - report MAE / RMSE / sMAPE honestly; skill vs baseline; next-week hotspot ranking overlap.

Run: .venv/bin/python src/05_forecast.py
"""

import sys
import json
import holidays 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import config as C

sns.set_theme(style="whitegrid", context="talk")
np.random.seed(C.SEED)
PRIMARY = "#1f4e79"; ACCENT = "#c0392b"; ORANGE = "#e67e22"; GREEN = "#27ae60"
TEST_WEEKS = 4


def smape(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    d = (np.abs(y) + np.abs(yhat))
    return np.mean(np.where(d == 0, 0, 2*np.abs(yhat - y)/d)) * 100


def savefig(fig, name):
    fig.tight_layout(); fig.savefig(C.FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"  [fig] {name}")


def main():
    conf = pd.read_parquet(C.CLEAN_PARQUET).query("is_confirmed").copy()
    conf["week_start"] = pd.to_datetime(conf["iso_week"].str.split("/").str[0])

    # weekly zone panel (zero-filled)
    wk = conf.groupby(["police_station", "week_start"]).size().reset_index(name="y")
    weeks = sorted(wk["week_start"].unique())
    # drop first & last week (partial coverage)
    keep_weeks = weeks[1:-1]
    print(f"weeks total={len(weeks)}  used (drop partial ends)={len(keep_weeks)}  "
          f"[{keep_weeks[0].date()} .. {keep_weeks[-1].date()}]")
    zones = wk["police_station"].unique()
    panel = (pd.MultiIndex.from_product([zones, keep_weeks], names=["police_station", "week_start"])
             .to_frame(index=False)
             .merge(wk, on=["police_station", "week_start"], how="left"))
    panel["y"] = panel["y"].fillna(0.0)
    panel = panel.sort_values(["police_station", "week_start"]).reset_index(drop=True)

    # leakage-safe features (all from PAST only)
    g = panel.groupby("police_station")["y"]
    for L in (1, 2, 3, 4):
        panel[f"lag{L}"] = g.shift(L)
    panel["roll4"] = g.shift(1).rolling(4, min_periods=1).mean().reset_index(level=0, drop=True)
    panel["roll4_std"] = g.shift(1).rolling(4, min_periods=1).std().reset_index(level=0, drop=True)
    week_index = {w: i for i, w in enumerate(keep_weeks)}
    panel["t"] = panel["week_start"].map(week_index)
    panel["month"] = panel["week_start"].dt.month
    panel["zone_code"] = panel["police_station"].astype("category").cat.codes

    # --- NEW AI FEATURES: HOLIDAYS & WEATHER ---
    print("    -> Injecting Holiday & Weather Data...")
    
    # 1. Holidays (India)
    in_holidays = holidays.India(years=[2023, 2024])
    def check_holiday(week_start):
        # Check if any day in this 7-day forecast window is a public holiday
        week_dates = [week_start + pd.Timedelta(days=i) for i in range(7)]
        return int(any(d in in_holidays for d in week_dates))
    
    panel["has_holiday"] = panel["week_start"].apply(check_holiday)
    
    # 2. Weather (Historical Temp Averages for Bengaluru Nov-Apr)
    # Note: Using deterministic historical averages here. In production, connect an Open-Meteo API.
    def mock_weather(month):
        temps = {11: 23.0, 12: 21.0, 1: 22.0, 2: 25.0, 3: 28.0, 4: 30.0}
        return temps.get(month, 25.0)
        
    panel["avg_temp"] = panel["month"].apply(mock_weather)
    # -------------------------------------------

    model_df = panel.dropna(subset=["lag1", "lag2", "lag3", "lag4"]).copy()
    test_weeks = keep_weeks[-TEST_WEEKS:]
    is_test = model_df["week_start"].isin(test_weeks)
    train, test = model_df[~is_test], model_df[is_test]
    print(f"train rows={len(train):,}  test rows={len(test):,}  (test weeks={len(test_weeks)})")

    # FIX: Added 'has_holiday' and 'avg_temp' to the feature list!
    FEATS = ["lag1", "lag2", "lag3", "lag4", "roll4", "roll4_std", "t", "month", "zone_code", "has_holiday", "avg_temp"]
    
    Xtr, ytr = train[FEATS].fillna(0), train["y"]
    Xte, yte = test[FEATS].fillna(0), test["y"]

    results = {}
    # --- baselines ---
    results["baseline_seasonal_naive"] = test["lag1"].values            # last week
    results["baseline_moving_avg4"] = test["roll4"].values              # 4-week MA
    # --- ML models ---
    gbm = HistGradientBoostingRegressor(loss="poisson", max_iter=400, max_depth=4,
                                        learning_rate=0.05, l2_regularization=1.0,
                                        random_state=C.SEED)
    gbm.fit(Xtr, ytr)
    results["HistGBM_poisson"] = np.clip(gbm.predict(Xte), 0, None)

    sc = StandardScaler().fit(Xtr)
    ridge = Ridge(alpha=10.0, random_state=C.SEED).fit(sc.transform(Xtr), ytr)
    results["Ridge"] = np.clip(ridge.predict(sc.transform(Xte)), 0, None)

    # --- metrics ---
    rows = []
    for name, pred in results.items():
        pred = np.nan_to_num(pred, nan=0.0)
        rows.append({
            "model": name,
            "MAE": round(mean_absolute_error(yte, pred), 2),
            "RMSE": round(np.sqrt(mean_squared_error(yte, pred)), 2),
            "sMAPE_%": round(smape(yte, pred), 1),
        })
    met = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    base_mae = met.loc[met.model == "baseline_seasonal_naive", "MAE"].iat[0]
    met["skill_vs_naive_%"] = ((base_mae - met["MAE"]) / base_mae * 100).round(1)
    met.to_csv(C.TBL_DIR / "forecast_metrics.csv", index=False)
    print("\n=== FORECAST METRICS (test = last 4 weeks) ===")
    print(met.to_string(index=False))

    best = met.iloc[0]["model"]
    print(f"\nbest model: {best}  (MAE {met.iloc[0]['MAE']} vs naive {base_mae}; "
          f"mean weekly zone load = {yte.mean():.0f})")

    # --- next-week hotspot ranking overlap (last test week) ---
    last_wk = test_weeks[-1]
    lw = test[test.week_start == last_wk].copy()
    lw["pred"] = results[best][test.week_start.values == last_wk]
    actual_top = set(lw.nlargest(10, "y")["police_station"])
    pred_top = set(lw.nlargest(10, "pred")["police_station"])
    overlap = len(actual_top & pred_top)
    rho, _ = spearmanr(lw["y"], lw["pred"])
    print(f"next-week hotspot ranking: top-10 overlap={overlap}/10  Spearman(rank)={rho:.2f}")

    # --- permutation importance (explainability) ---
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(gbm, Xte, yte, n_repeats=10, random_state=C.SEED, n_jobs=-1)
    imp = pd.DataFrame({"feature": FEATS, "importance": pi.importances_mean}).sort_values("importance")
    imp.to_csv(C.TBL_DIR / "forecast_feature_importance.csv", index=False)

    # ---------- FIGURES ----------
    print("\n[figures]")
    # MAE by model
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [GREEN if m == best else (PRIMARY if "baseline" not in m else "grey") for m in met["model"]]
    ax.barh(met["model"].iloc[::-1], met["MAE"].iloc[::-1], color=colors[::-1])
    ax.set_title(f"Weekly forecast error by model (best: {best})")
    ax.set_xlabel("MAE (violations/zone/week)")
    savefig(fig, "fig_forecast_mae.png")

    # feature importance
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp["feature"], imp["importance"], color=PRIMARY)
    ax.set_title("Forecast drivers — permutation importance (HistGBM)")
    ax.set_xlabel("mean MAE increase when shuffled")
    savefig(fig, "fig_forecast_importance.png")

    # actual vs predicted for top-4 zones over test weeks
    top4 = conf["police_station"].value_counts().head(4).index.tolist()
    test_pred = test.copy(); test_pred["pred"] = results[best]
    test_pred.to_csv(C.TBL_DIR / "forecast_predictions.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, z in zip(axes.ravel(), top4):
        d = test_pred[test_pred.police_station == z].sort_values("week_start")
        hist = model_df[(model_df.police_station == z) & (~model_df.week_start.isin(test_weeks))].sort_values("week_start")
        ax.plot(hist["week_start"], hist["y"], "o-", color="grey", label="actual (train)", ms=4)
        ax.plot(d["week_start"], d["y"], "o-", color=PRIMARY, label="actual (test)", ms=6)
        ax.plot(d["week_start"], d["pred"], "s--", color=ACCENT, label="forecast", ms=6)
        ax.set_title(z, fontsize=13); ax.tick_params(axis="x", rotation=30)
    axes.ravel()[0].legend(fontsize=10)
    fig.suptitle(f"Weekly violations: actual vs {best} forecast (top-4 zones)", fontsize=15)
    savefig(fig, "fig_forecast_zones.png")

    summary = {
        "weeks_used": len(keep_weeks),
        "test_weeks": TEST_WEEKS,
        "best_model": best,
        "best_MAE": float(met.iloc[0]["MAE"]),
        "naive_MAE": float(base_mae),
        "best_skill_vs_naive_pct": float(met.iloc[0]["skill_vs_naive_%"]),
        "best_sMAPE_pct": float(met.iloc[0]["sMAPE_%"]),
        "mean_weekly_zone_load": round(float(yte.mean()), 1),
        "next_week_top10_overlap": overlap,
        "next_week_rank_spearman": round(float(rho), 2),
        "top_feature": imp.iloc[-1]["feature"],
    }
    with open(C.TBL_DIR / "forecast_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[saved] forecast_summary.json -> {summary}")


if __name__ == "__main__":
    main()