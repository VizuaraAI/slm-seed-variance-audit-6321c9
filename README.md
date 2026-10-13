# slm-seed-variance-audit

Starter code for **"Does Seed Variance Exceed the Gaps in Published SLM
Comparisons?"** by Pradeep Adi (Vizuara SLM Bootcamp).

## The question

Small language model papers compare models on margins of one to three accuracy
points. Almost none of them report what the same model would have scored with a
different random seed. If run-to-run variance at this scale is larger than the
margins being reported, then a meaningful share of the claimed improvements in
the SLM literature cannot be distinguished from noise.

This repo measures that variance directly and expresses every published gap in
units of it, a quantity the paper calls **Seed-Equivalent Significance**:

```
SES = |reported gap| / sigma_seed
```

SES below 1 means the gap is smaller than the spread you get from re-running
the losing configuration with a different seed.

## What is in here

```
configs/default.yaml      the one file that defines a run; only `seed` varies
src/seeding.py            every source of randomness, in one place
src/data.py               benchmark + SFT loaders (everything downloads)
src/modeling.py           model loading and the 135M to 1.5B size ladder
src/train.py              fine-tune one model with one seed
src/evaluate.py           log-likelihood MCQ scoring over 3 prompt templates
src/claims.py             the hand-harvested database of published claims
src/analysis.py           sigma_seed, bootstrap CIs, SES, variance decomposition
src/plots.py              regenerates the paper's figures from your results
scripts/                  setup, smoke test, the sweep, the analysis
experiments/log_template.md  experiment log whose rows map to Table 1
```

Nothing large is committed. Every dataset downloads from the Hugging Face hub
on first use.

## Setup

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

Requires Python 3.10 or newer. A GPU makes the sweep comfortable but is not
required: the smoke test and the whole analysis path run on CPU.

## First three things to run

**1. Prove the pipeline works (a few minutes, CPU is fine).**

```bash
bash scripts/smoke_test.sh
```

Two seeds, 64 training examples, 40 eval items. The numbers are meaningless.
You are checking that training, scoring and analysis all run end to end before
you spend real compute.

**2. Run a real sweep on the smallest model.**

```bash
bash scripts/run_seed_sweep.sh EleutherAI/pythia-160m
```

Ten seeds, identical in every respect except the seed. Re-running the script
skips seeds that already finished, so an interruption costs you one seed.

**3. Look at the spread.**

```bash
bash scripts/run_analysis.sh
```

This prints the across-seed standard deviation with a bootstrap confidence
interval for every (model, benchmark, prompt) cell, and writes
`results/seed_stats.json`. That table is Table 1 of your paper.

The number to stare at is `sigma_seed`. Hold it next to the improvement margins
in any recent SLM paper. That comparison is the entire thesis.

## Then: the claim database

The experiments above measure the noise floor. The paper needs the other half:
the claims being measured against it.

```bash
cp experiments/claims_template.csv experiments/claims.csv
# fill in one row per published comparison, then:
python -m src.analysis
```

Harvesting rules are documented at the top of `src/claims.py`. Record numbers
exactly as printed, one row per comparison the authors draw a conclusion from,
and mark whether the paper reported any variance at all. That last column may
end up being the most quoted number in your paper.

## Climbing the ladder

```bash
for M in HuggingFaceTB/SmolLM2-135M EleutherAI/pythia-160m \
         EleutherAI/pythia-410m Qwen/Qwen2.5-0.5B EleutherAI/pythia-1.4b; do
  bash scripts/run_seed_sweep.sh "$M"
done
```

The hypothesis is that `sigma_seed` shrinks as parameter count grows. Five
rungs is enough to regress log sigma on log parameters and report a slope with
an interval.

## Honesty rules this repo enforces

- Seeds are contiguous integers chosen before you see results. Hand-picking a
  seed is the exact selection effect under study.
- Every run writes `resolved_config.yaml`. Two runs are only comparable if
  those files are identical except for the seed.
- Per-item correctness vectors are saved, not just accuracies, so paired
  bootstrap tests stay possible later.
- A gap below SES 1 means the evidence does not establish the claim. It does
  not mean the method does not work. Keep that distinction in every sentence
  you write.
