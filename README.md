# 🚦 ParkSight — AI Parking Intelligence
**Flipkart Gridlock 2.0 Hackathon Submission**

> 🏆 **Powered by MapmyIndia (Mappls)** | Built for Flipkart Last-Mile Logistics

**Turning 243k raw parking-violation tickets into a ranked, act-tomorrow enforcement map with proven business ROI.**

**ParkSight uses MapmyIndia-powered map visualization to identify high-impact Bengaluru traffic violation hotspots and convert them into enforcement recommendations.** It detects *where* illegal & spillover parking concentrates, models *how much* each hotspot likely chokes traffic flow, and outputs a **ranked list of enforcement zones** — so a city can replace blind, reactive patrols with data-targeted ones. 

It ships as a one-command local Streamlit app featuring an interactive MapmyIndia-backed dashboard, a GenAI tactical briefing, a Logistics ROI simulator, and live MapmyIndia API model validation.

> **Theme:** *Poor Visibility on Parking-Induced Congestion.*  
> **Problem:** *How can AI-driven parking intelligence detect illegal-parking hotspots and quantify their impact on traffic flow to enable targeted enforcement?*

---

## 🏆 Key Features Built for Gridlock 2.0

1. **MapmyIndia (Mappls) Deep Integration**:
   - **Hyper-Local Address Resolution:** Converts raw GPS coordinates into highly accurate Indian street addresses (e.g., "Koramangala 80ft Road Junction") using the Mappls Reverse Geocoding REST API.
   - **Live Traffic Routing Validation:** Proves our modeled congestion proxy actually works. Pings the MapmyIndia Routing API live to demonstrate a strong positive correlation between our AI Priority Score and real-world traffic delays.
2. **"What-If" ROI Simulator for Logistics**: Interactive slider to simulate deploying patrols to the top-N zones. Quantifies exact business value for delivery networks like Flipkart: last-mile delivery hours saved, fuel saved, and CO₂ reduced by clearing specific bottlenecks.
3. **The Priority Engine & GenAI Briefing**: Ranks the worst zones using an Enforcement Priority Index (Volume + Modeled Impact + Recurrence + Intensity). Generates a natural-language deployment briefing for traffic commanders.
4. **Patrol Schedule Optimizer**: Doesn't just tell you *where* to go, tells you *when*. Computes the optimal 3-hour patrol window for each zone based on peak day-of-week and hourly violation patterns.

---

## 🚀 Instructions to Run

> Prereqs: **Python 3.11**. The anonymized data is already bundled at `data/raw.csv` (and a precomputed `data/clean_violations.parquet` so the app works instantly). 

**One-command quickstart:**
```bash
# from the project root
./run.sh
```

That script: creates `.venv`, installs pinned deps, runs the full analysis pipeline (`src/01…08`), then launches the beautiful dark-themed UI at **http://localhost:8501**.

**Manual / step-by-step equivalent** (copy-pasteable):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_pipeline.py            # builds data/clean + outputs/ (tables, figures)
streamlit run app/streamlit_app.py    # opens the interactive demo on localhost:8501
```

*(Note: The application uses a securely configured MapmyIndia API key to power the hyper-local address resolution and live traffic validation.)*

---

## 🧠 Approach & Architecture

```
Traffic Violation Dataset
      ↓
Cleaning + Junction/Zone Mapping
      ↓
Severity + Congestion Impact Scoring
      ↓
MapmyIndia Routing API
      ↓
Route-Level Delay Impact Estimation
      ↓
What-If Simulator
      ↓
Patrol Recommendation Dashboard
```

**Why a *proxy* for congestion impact:** The raw dataset has **no traffic-flow signal** (no speed or delay). Rather than fabricate one, we model impact from three *data-derived* axes — offence severity, intersection exposure, and main-road exposure. **We then scientifically prove this proxy works** in the app by correlating our scores against real-time MapmyIndia routing data.

---

## 📊 Key Findings & ROI Numbers

| # | Finding | Metric / Proof |
|---|---|---|
| 1 | **Extreme spatial concentration** | Top 1% of ~250 m cells hold **33.9%** of all violations. |
| 2 | **Targeted ROI is massive** | Targeting just the top 5 zones clears **43%** of modeled citywide congestion impact. |
| 3 | **#1 Priority Zone is chronic** | **Upparpet** was active on **all 151 days** in the dataset (Score: 100/100). |
| 4 | **Optimal Patrol Windows** | E.g., deploying to Upparpet specifically between 08:00-11:00 captures **42%** of its weekly violations. |
| 5 | **Our model predicts real traffic** | Live MapmyIndia API validation shows strong correlation between our priority scores and actual live traffic delays. |
| 6 | **Evening enforcement blind spot** | Almost nothing is recorded 15:00–23:00 IST (**1.4%** of records) — a *visibility gap*, representing an untapped opportunity. |
| 7 | **Hotspots are predictable** | Next-week top-10 overlap is **9/10** — patrols can be reliably pre-positioned. |

---

## 🛡️ Honest Limitations
- **Congestion impact relies on a proxy model** — We mitigated this limitation by building the live MapmyIndia validation feature to prove the proxy strongly correlates with reality.
- **Hour-of-day reflects the enforcement-recording window**, not true diurnal demand (evenings are under-observed). We flag this evening gap as an operational finding.
- **No dwell-time** — we can't measure how long vehicles stayed based solely on the ticket timestamps.
- **Selection bias** — the data is *where enforcement looked*; absence of a ticket does not guarantee absence of a violation.

---

## 🛠️ Mapping & Routing Tech Stack
- **MapmyIndia Routing API**
  - Use cases: route-level impact estimation, last-mile delay calculation, what-if comparison for cleared zones, and enforcement planning support.

```text
Built with MapmyIndia APIs · pandas · scikit-learn · folium · plotly · Streamlit.
```
