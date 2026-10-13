# CLAUDE.md

Instructions for the coding agent working in this repository with Pradeep.

## What this project is

An empirical reproducibility study: **"Does Seed Variance Exceed the Gaps in
Published SLM Comparisons?"** We re-train small language models (135M to 1.5B)
across many random seeds under one fixed recipe, measure the across-seed
standard deviation of benchmark accuracy (`sigma_seed`), and compare that noise
floor against the improvement margins that published SLM papers report.

The headline statistic is **Seed-Equivalent Significance**,
`SES = |reported gap| / sigma_seed`. A gap below SES 1 is smaller than the
spread obtained by re-seeding the losing configuration.

Target: a short workshop paper (the Workshop on Insights from Negative Results
in NLP is the primary venue), plus an arXiv preprint. Timeline is 8 weeks of
part-time work.

## The one invariant

**Within a sweep, the only thing that may differ between runs is the integer
seed.** Everything in this codebase is organised around defending that claim to
a skeptical reviewer. Before changing anything, ask whether the change could
make two runs differ in some other way.

Concretely, if you are asked to add a feature:

- it must not introduce an unseeded source of randomness;
- it must not change behaviour based on wall-clock time, process ID, hostname,
  dict ordering, or filesystem iteration order;
- if it adds a knob, the knob goes in `configs/default.yaml` so it is captured
  in every run's `resolved_config.yaml`.

## Layout

| Path | Role |
| --- | --- |
| `configs/default.yaml` | the single source of truth for a run |
| `src/config.py` | config loading, `--set key=value` overrides, run dirs |
| `src/seeding.py` | all randomness control; treat as the measuring instrument |
| `src/data.py` | benchmark loaders normalised to `MultipleChoiceItem` |
| `src/modeling.py` | model/tokenizer loading, the size ladder |
| `src/train.py` | one process trains exactly one seed |
| `src/evaluate.py` | length-normalised log-likelihood MCQ scorer |
| `src/claims.py` | the harvested claim database and its schema |
| `src/analysis.py` | sigma, bootstrap CIs, SES, variance decomposition |
| `src/plots.py` | regenerates paper figures from `results/*.json` |
| `experiments/log_template.md` | log whose rows map to Table 1 of the paper |

Run artefacts live in `runs/<experiment>/<model>/seed<NNN>/` and contain
`resolved_config.yaml`, `train.json`, `eval.json`, and `checkpoint/`.
`results/` holds the aggregated JSON the paper reads from.

## Conventions

- Python 3.10+, standard library plus the pinned deps; no new heavy
  dependencies without asking.
- Modules are run as `python -m src.<module>`, each with an `argparse` CLI and
  a `main()`. Keep that shape.
- Every entry point takes `--config` and repeatable `--set key=value`.
- Plain PyTorch training loop on purpose. Do not replace it with `Trainer` or
  another framework: the loop is the measurement, so its randomness must stay
  visible on one screen.
- Imports of `torch` and `transformers` stay inside functions where practical,
  so that `src.analysis` and `src.claims` run on a laptop with no ML stack.
- Prefer small, readable functions over clever ones. Someone will audit this
  code, and the paper's credibility is the code's readability.
- No em dashes in comments, docstrings, or generated text.

## Statistics rules

- Standard deviations use `ddof=1`. Seeds are a sample, not a population.
- Any reported `sigma` gets a bootstrap confidence interval. With K=10 the
  point estimate is itself noisy and a reviewer will say so.
- When matching a `sigma` to a published gap, use the **largest** sigma
  observed for that benchmark. Conservative in the direction that makes our own
  thesis harder to support.
- Never report a mean without the spread. Never report a spread without K.

## Workflow with Pradeep

1. Start from a question written in `experiments/log_template.md`, not from
   code. If the question is not written down, ask for it.
2. Change one thing at a time. This project is about attributing variance, so
   a commit that changes two things is a commit that cannot be interpreted.
3. After any change to training or scoring, re-run `bash scripts/smoke_test.sh`
   before anything expensive.
4. Keep `results/` regenerable from `runs/` with one command. If the analysis
   stops being reproducible from the raw runs, stop and fix that first.

## Current state and next steps

Working: config, seeding, data loaders, training loop, MCQ scorer, analysis
(sigma, bootstrap, SES, two-way variance decomposition), plots, scripts.

Next, roughly in order:

1. Run the smoke test, then a real 10-seed sweep on `pythia-160m`.
2. Fill `experiments/claims.csv` with 120 to 200 harvested claims.
3. Add the kernel-nondeterminism ablation: repeat one seed N times with
   `train.deterministic_kernels` false, to find the floor beneath the floor.
4. Add the init-only vs data-order-only ablation (`train.shuffle_data` already
   isolates half of it).
5. Add `lm-evaluation-harness` cross-check parsing so harness scores land in
   the same `results/` schema as `src/evaluate.py` scores.
6. Add the log-sigma vs log-parameters regression with a confidence interval.
7. Add binomial confidence intervals on the fraction of claims below SES 1.

## Things to push back on

If Pradeep asks for any of these, say why it is a problem first:

- running fewer than 5 seeds for a headline number;
- comparing a multi-seed proposed configuration against a single-seed baseline;
- selecting the best checkpoint per run and reporting it as the run's score
  without saying so;
- quietly dropping a diverged run;
- changing the eval harness version mid-project without re-running everything.

Each of these is a practice the paper criticises. Doing one of them in our own
experiments would be the fastest way to lose the paper in review.
