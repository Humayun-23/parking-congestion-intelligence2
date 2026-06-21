#!/usr/bin/env bash
# One-command setup + pipeline + demo for Parking-Congestion Intelligence.
#   ./run.sh          -> create venv, install deps, run pipeline, launch the app
#   ./run.sh pipeline -> setup + run analysis pipeline only (no app)
#   ./run.sh app      -> launch the Streamlit app (assumes pipeline already ran)
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3.11}"
command -v "$PY" >/dev/null 2>&1 || PY="python3"

if [ ! -d ".venv" ]; then
  echo "==> creating virtual environment (.venv) with $PY"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> installing pinned dependencies"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

MODE="${1:-all}"

if [ "$MODE" = "all" ] || [ "$MODE" = "pipeline" ]; then
  echo "==> running analysis pipeline (01 -> 06)"
  python src/run_pipeline.py
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "app" ]; then
  echo "==> launching Streamlit app at http://localhost:8501"
  exec streamlit run app/streamlit_app.py
fi
