#!/usr/bin/env bash
# One-time setup. Creates a virtual environment and installs dependencies.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"

if [ ! -d ".venv" ]; then
  echo "[setup] creating .venv"
  "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir -p runs results figures
cp -n experiments/claims_template.csv experiments/claims.csv 2>/dev/null || true

echo
echo "[setup] done. Activate with:  source .venv/bin/activate"
echo "[setup] smoke test:           bash scripts/smoke_test.sh"
