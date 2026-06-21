# 🚦 ParkSight — Parking-Congestion Intelligence

**Turning 243k raw parking-violation tickets into a ranked, act-tomorrow enforcement map.**

ParkSight detects *where* illegal & spillover parking concentrates in Bengaluru, models *how
much* each hotspot likely chokes traffic flow, and outputs a **ranked list of enforcement zones**
— so a city can replace blind, reactive patrols with data-targeted ones. It ships as a one-command
local Streamlit app: an interactive heatmap + a live "Top Enforcement Zones" panel.

> Theme: *Poor Visibility on Parking-Induced Congestion.*
> Problem: *How can AI-driven parking intelligence detect illegal-parking hotspots and quantify
> their impact on traffic flow to enable targeted enforcement?*

---

## Instructions to Run

> Prereqs: **Python 3.11**. The anonymized data is already bundled at `data/raw.csv`
> (and a precomputed `data/clean_violations.parquet` so the app works instantly). To use a
> different file, replace `data/raw.csv` or set `PARKING_CSV=/abs/path/to.csv`. One command does
> setup + analysis + app.

```bash
# from the project root
./run.sh
```

That script: creates `.venv`, installs pinned deps, runs the full analysis pipeline
(`src/01…06`), then launches the demo at **http://localhost:8501**.

🚀 **Live Demo:** [https://parking-congestion-intelligence-gzqbam3yu4pfjp49fh5gsp.streamlit.app/](https://parking-congestion-intelligence-gzqbam3yu4pfjp49fh5gsp.streamlit.app/)

**Manual / step-by-step equivalent** (copy-pasteable):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_pipeline.py            # builds data/clean + outputs/ (tables, figures, metrics.json)
streamlit run app/streamlit_app.py    # opens the interactive demo on localhost:8501
```

Run only the analysis (no app): `./run.sh pipeline` · run only the app: `./run.sh app`.
Reproducible: deps are pinned in `requirements.txt`, seed fixed (`SEED=42`), no manual steps.

---

## What's in the box

```
parking-congestion-intelligence/
├── run.sh                      # one-command setup + pipeline + app
├── requirements.txt            # pinned deps (Python 3.11)
├── DATA_REPORT.md              # Step-0 recon: grain, quality, objective-support map
├── README.md                   # this file
├── src/
│   ├── config.py               # paths, seed, IST, severity weights, index weights
│   ├── 00_recon.py             # data recon (prints + recon_summary.json)
│   ├── 01_clean.py             # parse/clean/feature-build -> data/clean_violations.parquet
│   ├── 02_hotspots.py          # Obj 1 — DBSCAN + grid + zone/junction + temporal
│   ├── 03_congestion_proxy.py  # Obj 2 — modeled congestion-impact proxy + effect sizes
│   ├── 04_prioritization.py    # Obj 3 — Enforcement Priority Index (ranked output)
│   ├── 05_forecast.py          # Obj 4 — per-zone weekly forecast vs baselines (honest)
│   ├── 06_assemble.py          # roll-up -> outputs/metrics.json
│   └── run_pipeline.py         # runs 01->06
├── app/streamlit_app.py        # interactive local demo
└── outputs/
    ├── metrics.json            # single source of truth for every headline number
    ├── figures/                # 15 high-res PNG snapshots
    └── tables/                 # ranked zones/junctions, hotspots, forecast metrics, …
```

---

## Approach (architecture)

```
raw CSV (298k tickets)
   │  01_clean → parse JSON violation arrays · UTC→IST · de-dupe · drop rejected
   ▼            · per-event congestion-severity weight · snap to ~250 m grid
clean parquet (243k confirmed events)
   ├─ 02 HOTSPOTS      DBSCAN(haversine,60m) · 250m grid · zone & junction rollups · temporal
   ├─ 03 CONGESTION    impact = severity × (1 + 0.5·junction + 0.5·main-road)  ← MODELED proxy
   │                   effect sizes · Gini concentration · weighting sensitivity
   ├─ 04 PRIORITIZE    EPI = 0.30·volume + 0.35·impact + 0.20·recurrence + 0.15·intensity
   │                   → ranked top-N zones + junctions + recommended patrol windows
   └─ 05 FORECAST      weekly zone panel · leakage-safe lags · HistGBM/Ridge vs naive/MA baselines
   ▼
metrics.json → Streamlit app (map + filters + Top Enforcement Zones + forecast)
```

**Why a *proxy* for congestion impact:** the data has **no traffic-flow signal** (no speed,
volume, delay, or occupancy). Rather than fabricate one, we model impact from three *data-derived*
axes — offence severity, intersection exposure, and main-road exposure — label it clearly as a
proxy, and prove the **zone ranking is robust to the weighting** (Spearman ≥ 0.98 vs alternatives).

---

## Key findings (every number from the data)

| # | Finding | Number |
|---|---|---|
| 1 | Confirmed parking-violation events analysed (Nov '23–Apr '24) | **243,274** (of 298k; rejected/dupes removed) |
| 2 | **Extreme spatial concentration** — top 1% of ~250 m cells hold a third of all violations | **33.9%** in ~33 cells; grid Gini **0.86** |
| 3 | **Targeting the top 5 zones addresses nearly half the modeled congestion impact** | top-5 = **43%**, top-10 = **60%**, top-15 = **71%** (of 54 zones) |
| 4 | **#1 priority zone** (chronic — active on **all 151 days** in the window) | **Upparpet** (29,200 violations, score 100/100) |
| 5 | **#1 priority junction** | **Safina Plaza Jn**; top-10 junctions = **48%** of junction impact |
| 6 | Counter-intuitive: **heavy-impact parking sits on open carriageways, not junctions** | high-severity **13.5%** off-junction vs **4.2%** at junction (**3.2×**) |
| 7 | **Evening enforcement blind spot** — almost nothing is recorded 15:00–23:00 IST | **1.4%** of records (a *visibility gap*, not zero demand) |
| 8 | **Hotspots persist week-to-week → patrols can be pre-positioned** | next-week top-10 overlap **9/10**, rank Spearman **0.83** |
| 9 | Honest forecast caveat: a 4-week moving average beats ML on counts | MAE **72** vs naive 79 (sMAPE 41%) on 21 weeks |

**Dominant offence:** WRONG PARKING & NO PARKING (these two labels alone appear on the large
majority of tickets). **Dominant vehicle:** SCOOTER, then CAR. **Weekends over-index** (Sunday is
the single busiest day).

---

## Honest limitations
- **Congestion impact is *modeled*, not measured** — no flow data in the source. Validation against
  a real traffic feed is the #1 next step (join design is ready: zone × hour).
- **Hour-of-day reflects the enforcement-recording window**, not true diurnal demand (evenings are
  under-observed). We use it operationally and flag the evening gap as a finding.
- **No dwell-time** (`closed_datetime` is 100% null) — we can't measure how long vehicles stayed.
- **Selection bias** — the data is *where enforcement looked*; absence ≠ no violation.

See [DATA_REPORT.md](DATA_REPORT.md) for the full recon and quality assessment.
```text
Built with pandas · numpy · scikit-learn · scipy · folium · plotly · Streamlit. Local-only. Seed = 42.
```
