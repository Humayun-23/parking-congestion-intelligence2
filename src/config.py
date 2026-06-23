"""
Central configuration: paths, seeds, time zone, severity weights, grid/DBSCAN params,
and the priority-index weights. Imported by every other module so the whole pipeline
shares one source of truth.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ----------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# Load environment variables from the root .env file
load_dotenv(ROOT / ".env")

def _resolve_raw_csv():
    """Find the raw CSV. Priority: $PARKING_CSV -> data/raw.csv -> data/*.csv ->
    the original Downloads location used during development."""
    env = os.environ.get("PARKING_CSV")
    if env and Path(env).exists():
        return env
    local = DATA_DIR / "raw.csv"
    if local.exists():
        return str(local)
    globbed = sorted(DATA_DIR.glob("*violation*.csv")) or sorted(DATA_DIR.glob("*.csv"))
    if globbed:
        return str(globbed[0])
    return "/Users/arpitdhankani/Downloads/jan to may police violation_anonymized791b166(1).csv"


RAW_CSV = _resolve_raw_csv()
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
CLEAN_PARQUET = DATA_DIR / "clean_violations.parquet"
METRICS_JSON = OUT_DIR / "metrics.json"
for _d in (DATA_DIR, OUT_DIR, FIG_DIR, TBL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# API Keys
MAPMYINDIA_API_KEY = os.getenv("MAPMYINDIA_API_KEY", "ssajusqohbvsidqzyqfyqeksiolttnlywpsg")

# ------------------------------------------------------------------- reproducibility
SEED = 42

# --------------------------------------------------------------------------- geo / tz
IST = "Asia/Kolkata"
# Bengaluru bounding box (data is 100% inside this)
BBOX = dict(lat_min=12.6, lat_max=13.3, lon_min=77.3, lon_max=77.9)
EARTH_RADIUS_M = 6_371_000.0

# Spatial grid for fine-grained hotspot binning (independent of admin boundaries).
# At Bengaluru's latitude 1 deg lat ~= 110.6 km, 1 deg lon ~= 108.4 km.
GRID_M = 250.0
M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON = 108_400.0  # ~ 111320 * cos(12.97 deg)

# DBSCAN micro-hotspot params (haversine). eps in metres -> radians at runtime.
DBSCAN_EPS_M = 60.0
DBSCAN_MIN_SAMPLES = 25

# --------------------------------------------------------- congestion severity weights
CONGESTION_WEIGHTS = {
    # --- high: blocks live carriageway / intersection ---
    "PARKING IN A MAIN ROAD": 3,
    "DOUBLE PARKING": 3,
    "PARKING NEAR ROAD CROSSING": 3,
    "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS": 3,
    "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE": 3,
    "OBSTRUCTING DRIVER": 3,
    "AGAINST ONE WAY/NO ENTRY": 3,
    # --- medium: occupies road space / capacity loss ---
    "WRONG PARKING": 2,
    "NO PARKING": 2,
    "PARKING OTHER THAN BUS STOP": 2,
    "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC": 2,
    "H T V PROHIBITED": 2,
    # --- low: footpath / pedestrian, minor flow impact ---
    "PARKING ON FOOTPATH": 1,
    # --- zero: administrative / moving / fare violations (no parking-congestion impact) ---
    "DEFECTIVE NUMBER PLATE": 0,
    "USING BLACK FILM/OTHER MATERIALS": 0,
    "WITHOUT SIDE MIRROR": 0,
    "REFUSE TO GO FOR HIRE": 0,
    "DEMANDING EXCESS FARE": 0,
    "FAIL TO USE SAFETY BELTS": 0,
    "RIDER NOT WEARING HELMET": 0,
    "2W/3W - USING MOBILE PHONE": 0,
    "OTHER - USING MOBILE PHONE": 0,
    "CARRYING LENGHTY MATERIAL": 0,
    "JUMPING TRAFFIC SIGNAL": 0,
    "U TURN PROHIBITED": 0,
    "STOPING ON WHITE/STOP LINE": 0,
    "VIOLATING LANE DISIPLINE": 0,
}
DEFAULT_WEIGHT = 2  # unseen labels default to "medium" (most are parking)

# Labels we treat as genuine parking offences (weight >= 1). Used for the parking subset.
NON_PARKING_LABELS = {k for k, v in CONGESTION_WEIGHTS.items() if v == 0}

IMPACT_JUNCTION_BOOST = 0.50
IMPACT_MAINROAD_BOOST = 0.50

# Validation statuses to drop from the "confirmed" analysis set.
DROP_VALIDATION = {"rejected", "duplicate"}

# --------------------------------------------------- enforcement priority index weights
PRIORITY_WEIGHTS = {
    "violation_volume": 0.30,      # how many confirmed violations
    "congestion_impact": 0.35,     # severity-weighted, intersection-exposed load
    "recurrence": 0.20,            # chronic: distinct active days
    "spatial_intensity": 0.15,     # concentration (events per active grid cell)
}

TOP_N_ZONES = 15
TOP_N_JUNCTIONS = 20


def grid_id(lat, lon):
    """Snap a coordinate to the SW corner of its ~GRID_M cell; returns (glat, glon)."""
    import numpy as np
    dlat = GRID_M / M_PER_DEG_LAT
    dlon = GRID_M / M_PER_DEG_LON
    glat = np.floor(lat / dlat) * dlat
    glon = np.floor(lon / dlon) * dlon
    return np.round(glat, 6), np.round(glon, 6)