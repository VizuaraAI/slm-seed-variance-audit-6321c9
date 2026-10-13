#!/usr/bin/env bash
# Smallest possible end-to-end pass: two seeds, a tiny model, a few hundred
# examples. Runs on CPU in a few minutes. The numbers are meaningless; the
# point is to prove the pipeline runs before you spend GPU hours on it.
set -euo pipefail

cd "$(dirname "$0")/.."

for SEED in 0 1; do
  echo "[smoke] seed $SEED"
  python -m src.train \
    --seed "$SEED" \
    --set experiment.name=smoke \
    --set data.sft_max_examples=64 \
    --set data.max_seq_length=128 \
    --set train.batch_size=4 \
    --set train.log_every=2

  python -m src.evaluate \
    --seed "$SEED" \
    --model-path "runs/smoke/EleutherAI__pythia-160m/seed$(printf '%03d' "$SEED")/checkpoint" \
    --set experiment.name=smoke \
    --set eval.tasks='["arc_easy"]' \
    --set eval.max_eval_examples=40 \
    --set eval.prompt_templates='["plain","qa"]'
done

python -m src.analysis \
  --set experiment.name=smoke \
  --skip-claims \
  --out results/smoke

echo
echo "[smoke] pipeline works end to end. Now run scripts/run_seed_sweep.sh."
