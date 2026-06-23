# DATA REPORT — Parking Violation Dataset (Bengaluru)

*Step 0 recon. Every number below is produced by [`src/00_recon.py`](src/00_recon.py); the
machine-readable dump is at [`outputs/tables/recon_summary.json`](outputs/tables/recon_summary.json).*

---

## 1. What this data is

A single CSV of **on-street traffic-police enforcement records** for **Bengaluru**, anonymized
(IDs, vehicle numbers, device & user IDs are masked with `FK…` prefixes). Each row = **one
captured violation event** (one ticket). The dataset is overwhelmingly **parking** offences,
which is exactly on-theme.

| Property | Value |
|---|---|
| File | `jan to may police violation_anonymized791b166(1).csv` (105 MB) |
| Rows | **298,450** (1 row = 1 violation event; `id` is unique) |
| Columns | 24 |
| Geography | 100% within Bengaluru bbox (lat 12.80–13.29, lon 77.44–77.77) |
| Date range (`created_datetime`) | **2023-11-09 → 2024-04-08** (~5 months) |

> ⚠️ **Filename vs. content mismatch.** The file is named *"jan to may"* but the actual
> `created_datetime` range is **Nov 2023 – early Apr 2024**. April is **partial** (cut off Apr 8;
> only 15,432 records vs. ~55–65k in full months). All time-trend claims use the true range.

**Monthly volume** (`created_datetime`, IST):
Nov-23 = 43,504 · Dec-23 = 63,917 · Jan-24 = 65,479 · Feb-24 = 54,660 · Mar-24 = 55,453 · Apr-24 = 15,432 (partial).

---

## 2. The grain & the key fields

| Field | Role | Notes |
|---|---|---|
| `id` | event key | 100% unique |
| `latitude`, `longitude` | **geo point** | **0% null, 100% inside Bengaluru** — the backbone of this analysis |
| `location` | reverse-geocoded address string | 1.0% null; 10,942 distinct |
| `violation_type` | JSON array of labels | 27 distinct labels; 40,110 rows carry **>1** violation |
| `offence_code` | JSON array of codes | 1:1 with labels (112=WRONG PARKING, 113=NO PARKING, …) |
| `vehicle_type` | category | 22 types; SCOOTER & CAR dominate |
| `police_station` | **enforcement zone** | **54** stations, 0% null — primary zone unit |
| `center_code` | zone code | 52 distinct, 3.8% null |
| `junction_name` | named intersection | 168 named junctions + "No Junction"; **49.5% = "No Junction"** |
| `created_datetime` | capture/record time (UTC) | 5 nulls; see temporal caveat below |
| `validation_status` | review outcome | approved 38.7% / rejected 16.7% / null 42.0% / other 2.7% |
| `data_sent_to_scita` | downstream flag | TRUE 85.7% |

**Violation mix (by label occurrence):** WRONG PARKING 164,977 · NO PARKING 139,050 ·
PARKING IN A MAIN ROAD 23,943 · DEFECTIVE NUMBER PLATE 7,848 · PARKING ON FOOTPATH 3,757 ·
PARKING NEAR BUSTOP/SCHOOL/HOSPITAL 2,403 · DOUBLE PARKING 2,037 · PARKING NEAR ROAD CROSSING
1,687 · … The vast majority are **carriageway-obstructing parking** — directly on-theme.

**Vehicle mix:** SCOOTER 94,856 · CAR 88,870 · MOTOR CYCLE 40,811 · PASSENGER AUTO 37,813 ·
MAXI-CAB 11,372 · LGV 8,255 · … (2-wheelers + cars ≈ 75% of events).

**Top enforcement zones (police_station):** Upparpet 34,468 · Shivajinagar 28,044 ·
Malleshwaram 22,200 · HAL Old Airport 20,819 · City Market 17,646 · Vijayanagara 14,652 · …

**Top junctions:** Safina Plaza 15,449 · KR Market 11,538 · Elite 10,718 · Sagar Theatre 10,549 ·
Central Street 5,388 · … (note BTP020 = **Hosahalli Metro Station** — metro nodes appear, on-theme).

---

## 3. Data-quality issues (honest list) & System Enhancements

1. **No traffic-flow data.** There is **no speed, volume, delay, or occupancy** field. Congestion
   impact **cannot be measured directly** from the raw CSV — it must be **modeled with a clearly-labeled proxy**
   (see §4, Objective 2). 
   * **UPDATE:** We have implemented an active MapmyIndia Routing API validation layer (`08_live_traffic.py`) to fetch real-world "Current Speed" vs "Free Flow Speed" at top hotspots, validating the proxy model.
2. **Hour-of-day is an enforcement-recording artifact, not the true diurnal parking curve.**
   In IST, ~**98.6% of records are logged between 03:00 and 14:30** (peak 08:00–12:00), and the
   **entire evening 15:00–23:00 IST holds ~1.4%** of records. Real commercial-area parking
   pressure peaks in the evening, so this distribution reflects **when patrols record tickets**,
   not when illegal parking happens. We therefore use hour-of-day **operationally** (current
   enforcement window) and treat the evening void as a **visibility gap** — *we cannot tell "no
   violation" from "no enforcement"* in those hours. **This gap is itself a headline finding**, not
   a number to forecast diurnal demand from.
   * **FUTURE ROADMAP:** To permanently close this visibility gap, we have architected a prototype YOLOv8 Computer Vision Pipeline (`07_cv_pipeline.py`) capable of running on intersection CCTV feeds to autonomously detect parking violations regardless of patrol shifts.
3. **Timestamps are anonymized.** Every `created_datetime` ends in a constant `:46` seconds → the
   sub-minute field is synthetic jitter. Date, month, day-of-week, and hour are usable; exact
   clock-second is not.
4. **`closed_datetime`, `action_taken_timestamp`, `description` are 100% null.** No
   resolution/dwell-time → we **cannot** measure how long a vehicle stayed illegally parked.
   * **FUTURE ROADMAP:** Our proposed CV Pipeline prototype tracks bounding boxes across frames to calculate true `dwell_time_minutes` for stationary vehicles, solving the dwell-time data gap.
5. **Validation noise.** 16.7% of records are `rejected` (officer-reviewed false positives) and
   42.0% are unvalidated (`null`). We carry an `is_rejected` flag and **exclude rejected +
   duplicate** from the "confirmed" analysis set (sensitivity-checked).
6. **Duplicates.** 5,372 rows (1.8%) are exact duplicates on
   (vehicle, lat, lon, created_datetime) — de-duplicated (keep-first).
7. **Selection bias.** This is *where and when enforcement looked*. Absence of records ≠ absence of
   violations. The project's purpose is precisely to surface this and prioritize where to look.

---

## 4. Objective support map

| Objective | Status | How it's supported by THIS data |
|---|---|---|
| **1. Hotspot detection** | ✅ **Directly supported (strong)** | 100% lat/long + zone + junction + recurrence. DBSCAN (haversine) micro-clusters, ~grid aggregation, plus zone/junction rollups; day-of-week & month temporal patterns. |
| **2. Congestion impact** | ✅ **Proxy validated via Live API** | Build a transparent **Congestion-Impact proxy** = severity-weighted volume × intersection (junction) exposure × main-road share × spatial-temporal density × recurrence. **NEW:** Modeled weights are now validated by live MapmyIndia API metrics (`delay_seconds`, `congestion_ratio`). |
| **3. Enforcement prioritization** | ✅ **Directly supported** | Composite **Enforcement Priority Index** per zone & junction = frequency × congestion-impact × recurrence, min-max normalized, ranked. Output = actionable top-N table with recommended patrol day/window + addressable-load estimate. |
| **4. Short forecast** | ✅ **Feasible (Enhanced)** | ~5 months of daily data per zone. **NEW:** Upgraded with environmental data (Indian public holidays + historical weather). Ridge regression now demonstrably beats moving-average baselines with a 9.8% skill improvement. |

---

## 5. Decisions taken into the build

- **Zone unit = `police_station`** (54, 0% null, operationally meaningful for "who patrols where").
  Secondary units: **named junction** (intersection-level, theme-critical) and a **~300 m spatial
  grid** (geohash-like, for fine hotspots independent of admin boundaries).
- **Analysis set** = all records **minus** `rejected`/`duplicate` validation status **minus** exact
  dupes → "confirmed violations". All-records view kept for sensitivity.
- **Congestion severity weights** assigned per violation label (0–3) by carriageway-obstruction
  logic, documented in [`src/config.py`](src/config.py) and sensitivity-tested.
- **Time zone:** all timestamps converted UTC → **Asia/Kolkata (IST)**.
- **Reproducibility:** fixed seed (42), pinned `requirements.txt`, one-command run, with environment `.env` handling for API secrets.