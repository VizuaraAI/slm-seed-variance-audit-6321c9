#!/usr/bin/env bash
# Independent cross-check with lm-evaluation-harness.
#
# Our own scorer (src/evaluate.py) exists because we need per-item outputs and
# prompt-template control. The harness exists because reviewers trust it. If
# the two disagree by more than a published gap, that disagreement goes in the
# paper: it is evidence about protocol sensitivity, not a bug to hide.
#
# Install first:  pip install "lm-eval>=0.4.5"
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_PATH="${1:?usage: bash scripts/run_harness_eval.sh <checkpoint-dir> [seed]}"
SEED="${2:-0}"
TASKS="${TASKS:-arc_easy,hellaswag}"
OUT="results/harness/seed$(printf '%03d' "$SEED")"

mkdir -p "$OUT"

# Pin the harness version in the paper's appendix. Evaluation results are not
# comparable across harness versions, which is the point of recording it.
python -c "import lm_eval; print('[harness] lm-eval', lm_eval.__version__)"

lm_eval \
  --model hf \
  --model_args "pretrained=$MODEL_PATH" \
  --tasks "$TASKS" \
  --batch_size 8 \
  --seed "$SEED" \
  --output_path "$OUT"

echo "[harness] wrote $OUT"
