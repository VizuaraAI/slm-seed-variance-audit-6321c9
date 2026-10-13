#!/usr/bin/env bash
# The main experiment: train and evaluate one model across every seed.
#
# Usage:
#   bash scripts/run_seed_sweep.sh                         # config defaults
#   bash scripts/run_seed_sweep.sh EleutherAI/pythia-410m   # one ladder rung
#   SEEDS="0 1 2" bash scripts/run_seed_sweep.sh            # fewer seeds
#
# One process per seed, so a crash costs you one seed and not the sweep.
# Runs are skipped if their eval.json already exists, which makes this script
# safe to re-run after an interruption.
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${1:-EleutherAI/pythia-160m}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
EXP="${EXP:-seed-variance-audit}"
SAFE_MODEL="${MODEL//\//__}"

echo "[sweep] model=$MODEL seeds=[$SEEDS] experiment=$EXP"

for SEED in $SEEDS; do
  RUN_DIR="runs/$EXP/$SAFE_MODEL/seed$(printf '%03d' "$SEED")"

  if [ -f "$RUN_DIR/eval.json" ]; then
    echo "[sweep] seed $SEED already complete, skipping"
    continue
  fi

  echo "[sweep] ===== seed $SEED ====="
  python -m src.train \
    --seed "$SEED" \
    --set experiment.name="$EXP" \
    --set model.name="\"$MODEL\""

  python -m src.evaluate \
    --seed "$SEED" \
    --model-path "$RUN_DIR/checkpoint" \
    --set experiment.name="$EXP" \
    --set model.name="\"$MODEL\""
done

echo
echo "[sweep] done. Summarise with:"
echo "  python -m src.analysis --set experiment.name=$EXP"
