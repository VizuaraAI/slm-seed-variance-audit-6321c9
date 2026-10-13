#!/usr/bin/env bash
# Turn the runs on disk into the paper's table and figures.
# Cheap and CPU-only, so run it as often as you like.
set -euo pipefail

cd "$(dirname "$0")/.."

EXP="${EXP:-seed-variance-audit}"

python -m src.analysis --set experiment.name="$EXP" --out results
python -m src.plots --results results --figures figures

echo
echo "[analysis] results/seed_stats.json           -> Table 1 of the paper"
echo "[analysis] results/variance_decomposition.json -> the prompt-vs-seed ablation"
echo "[analysis] results/claim_audit.json          -> Figure 2b and the headline"
echo "[analysis] figures/*.png                     -> copy into paper/figures/"
