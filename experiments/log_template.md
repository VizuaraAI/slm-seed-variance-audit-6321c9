# Experiment log

One entry per sweep. Copy the block below, fill it in while the run is going,
not afterwards from memory. The columns map one-to-one onto Table 1 of the
paper, so a complete log means the results table writes itself.

A variance paper lives or dies on whether a reader believes two runs really
were identical except for the seed. This log is the evidence for that claim.

---

## Entry template (copy this)

**Entry ID:** EXP-00
**Date:** YYYY-MM-DD
**Question this run answers (one sentence):**

**Status:** planned / running / done / abandoned

### Configuration

| Field | Value |
| --- | --- |
| Model | e.g. EleutherAI/pythia-160m |
| Parameters (M) | measured, not from the model card |
| Regime | fine-tune / pretrain-from-scratch |
| SFT dataset and size | |
| Seeds | e.g. 0-9 |
| Epochs / LR / batch / accum | |
| Deterministic kernels | true / false |
| Benchmarks | e.g. arc_easy, hellaswag |
| Prompt templates | e.g. plain, qa, instruct |
| Eval items per task | |
| Config hash / commit | git rev-parse --short HEAD |
| Hardware | |

### Results (maps to Table 1 of the paper)

| Benchmark | Template | Seeds | Mean acc. (%) | sigma_seed | 95% CI on sigma | Range (max-min) |
| --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |

### Variance decomposition (maps to the ablation section)

| Benchmark | sigma_seed | sigma_prompt | sigma_residual | Which dominates |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

### Claim audit touched by this run (maps to Figure 2b)

| Claim ID | Paper | Benchmark | Reported gap | sigma_seed used | SES | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |

### What actually happened

- Wall clock per seed:
- Failures, restarts, OOMs:
- Anything that was NOT identical across seeds (be honest, this is the one
  that matters):

### What I concluded, and what I will do next

-
-

---

## Running index

| Entry | Date | Model | Seeds | Headline number | Status |
| --- | --- | --- | --- | --- | --- |
| EXP-00 | | | | | planned |

---

## Standing checklist before you trust a sweep

- [ ] `resolved_config.yaml` in every run directory is byte-identical except
      for `_seed`.
- [ ] Seed list is contiguous and was fixed before you saw any results.
- [ ] No run was dropped. If one was, the reason is recorded above.
- [ ] Baseline got the same number of seeds as the proposed configuration.
- [ ] Harness version and prompt template are recorded.
- [ ] Per-seed scores are saved, not just the mean.
