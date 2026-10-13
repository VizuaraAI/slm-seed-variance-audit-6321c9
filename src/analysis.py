"""The statistics that turn a pile of runs into the paper's claim.

Three quantities matter.

sigma_seed
    The standard deviation of a benchmark score across seeds, holding the
    recipe fixed. Computed with ddof=1, because the seeds are a sample from
    the population of possible seeds, not the whole population.

Seed-Equivalent Significance (SES)
    SES = |reported gap| / sigma_seed. The unit is "seed standard deviations".
    SES below 1 means the published improvement is smaller than the spread you
    get from re-running the loser with a different seed. SES above 2 means the
    gap would survive the replication. This is the paper's headline statistic
    and the x axis of figure 2b.

Variance decomposition
    Total score variance split into a seed component and a prompt-format
    component. If prompt variance dominates seed variance, the honest headline
    changes from "seeds are the problem" to "the evaluation protocol is the
    problem", and that is a finding, not a failure.

Everything is NumPy and SciPy only, so this file runs on a laptop with no GPU
and no model downloads: useful, because you will rerun the analysis far more
often than you rerun the training.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import claims as claims_mod
from . import config as config_mod


@dataclass
class SeedStats:
    """Across-seed summary for one (model, task, template) cell."""

    model: str
    task: str
    template: str
    n_seeds: int
    mean: float
    std: float
    minimum: float
    maximum: float
    std_ci_low: float
    std_ci_high: float
    scores: List[float]

    @property
    def spread(self) -> float:
        """Max minus min: the number a reader intuitively reacts to."""
        return self.maximum - self.minimum

    def row(self) -> Dict[str, object]:
        return {
            "model": self.model,
            "task": self.task,
            "template": self.template,
            "n_seeds": self.n_seeds,
            "mean": round(self.mean, 2),
            "std": round(self.std, 3),
            "spread": round(self.spread, 2),
            "std_ci_low": round(self.std_ci_low, 3),
            "std_ci_high": round(self.std_ci_high, 3),
        }


def load_runs(run_root: str) -> List[Dict]:
    """Collect every eval.json under a run root."""
    pattern = os.path.join(run_root, "**", "eval.json")
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise FileNotFoundError(
            f"no eval.json found under {run_root}; run the seed sweep first"
        )
    payloads = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["_path"] = path
        payloads.append(payload)
    return payloads


def score_table(payloads: Sequence[Dict]) -> Dict[Tuple[str, str, str], Dict[int, float]]:
    """Index accuracies as (model, task, template) -> {seed: accuracy}."""
    table: Dict[Tuple[str, str, str], Dict[int, float]] = {}
    for payload in payloads:
        model = payload.get("base_model") or payload["model"]
        seed = int(payload["seed"])
        for result in payload["results"]:
            key = (model, result["task"], result["template"])
            cell = table.setdefault(key, {})
            if seed in cell:
                raise ValueError(
                    f"duplicate seed {seed} for {key}; two runs wrote the same "
                    "cell, so one of them is not the run you think it is"
                )
            cell[seed] = float(result["accuracy"])
    return table


def bootstrap_std_ci(
    scores: Sequence[float],
    resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 12345,
) -> Tuple[float, float]:
    """Percentile bootstrap interval for the standard deviation.

    With 10 seeds the point estimate of sigma is itself noisy, and a reviewer
    will say so. Reporting the interval answers the objection before it is
    raised. The bootstrap RNG is seeded, which is the one place in this project
    where a fixed seed is simply good hygiene rather than the object of study.
    """
    values = np.asarray(scores, dtype=float)
    if values.size < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(resamples, values.size), replace=True)
    stds = draws.std(axis=1, ddof=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(stds, alpha)),
        float(np.quantile(stds, 1.0 - alpha)),
    )


def summarise_cell(
    model: str,
    task: str,
    template: str,
    seed_scores: Dict[int, float],
    resamples: int = 10000,
    confidence: float = 0.95,
) -> SeedStats:
    seeds = sorted(seed_scores)
    values = np.array([seed_scores[s] for s in seeds], dtype=float)
    low, high = bootstrap_std_ci(values, resamples, confidence)
    return SeedStats(
        model=model,
        task=task,
        template=template,
        n_seeds=values.size,
        mean=float(values.mean()),
        std=float(values.std(ddof=1)) if values.size > 1 else 0.0,
        minimum=float(values.min()),
        maximum=float(values.max()),
        std_ci_low=low,
        std_ci_high=high,
        scores=[float(v) for v in values],
    )


def summarise_all(
    payloads: Sequence[Dict], resamples: int = 10000, confidence: float = 0.95
) -> List[SeedStats]:
    table = score_table(payloads)
    return [
        summarise_cell(model, task, template, cell, resamples, confidence)
        for (model, task, template), cell in sorted(table.items())
    ]


def variance_decomposition(
    payloads: Sequence[Dict], model: str, task: str
) -> Dict[str, float]:
    """Split variance into seed and prompt-format components.

    A two-way layout with seeds as rows and templates as columns, analysed as a
    simple random-effects decomposition. Components are clipped at zero: a
    negative variance estimate is an artefact of small samples, not a negative
    variance.
    """
    table = score_table(payloads)
    templates = sorted(
        t for (m, tk, t) in table if m == model and tk == task
    )
    if not templates:
        raise KeyError(f"no cells for model={model!r} task={task!r}")

    seed_sets = [set(table[(model, task, t)]) for t in templates]
    shared_seeds = sorted(set.intersection(*seed_sets))
    if len(shared_seeds) < 2:
        raise ValueError(
            "need at least 2 seeds present under every template; "
            f"found {len(shared_seeds)}"
        )

    matrix = np.array(
        [[table[(model, task, t)][s] for t in templates] for s in shared_seeds],
        dtype=float,
    )
    n_seeds, n_templates = matrix.shape
    grand = matrix.mean()
    seed_means = matrix.mean(axis=1)
    template_means = matrix.mean(axis=0)

    ss_seed = n_templates * float(((seed_means - grand) ** 2).sum())
    ss_template = n_seeds * float(((template_means - grand) ** 2).sum())
    residual = matrix - seed_means[:, None] - template_means[None, :] + grand
    ss_residual = float((residual**2).sum())

    df_seed = n_seeds - 1
    df_template = n_templates - 1
    df_residual = max(1, df_seed * df_template)
    ms_seed = ss_seed / df_seed
    ms_template = ss_template / max(1, df_template)
    ms_residual = ss_residual / df_residual

    var_seed = max(0.0, (ms_seed - ms_residual) / n_templates)
    var_template = max(0.0, (ms_template - ms_residual) / n_seeds)
    return {
        "model": model,
        "task": task,
        "n_seeds": n_seeds,
        "n_templates": n_templates,
        "sigma_seed": round(float(np.sqrt(var_seed)), 3),
        "sigma_prompt": round(float(np.sqrt(var_template)), 3),
        "sigma_residual": round(float(np.sqrt(ms_residual)), 3),
        "grand_mean": round(float(grand), 2),
    }


def seed_equivalent_significance(
    gap: float, sigma_seed: float, floor: float = 1.0, strong: float = 2.0
) -> Dict[str, object]:
    """Express one published gap in units of seed standard deviation."""
    if sigma_seed <= 0:
        return {"ses": float("inf"), "verdict": "no seed variance measured"}
    ses = abs(gap) / sigma_seed
    if ses < floor:
        verdict = "within seed noise"
    elif ses < strong:
        verdict = "ambiguous"
    else:
        verdict = "survives replication"
    return {"ses": float(ses), "verdict": verdict}


def audit_claims(
    claim_list: Sequence[claims_mod.Claim],
    stats: Sequence[SeedStats],
    floor: float = 1.0,
    strong: float = 2.0,
) -> List[Dict[str, object]]:
    """Score every harvested claim against the measured seed noise.

    sigma_seed is matched per benchmark, using the largest sigma observed for
    that benchmark across prompt templates. That choice is conservative in the
    direction that makes the paper's own claim harder to make, which is the
    direction you want to be conservative in.
    """
    by_task: Dict[str, float] = {}
    for stat in stats:
        by_task[stat.task] = max(by_task.get(stat.task, 0.0), stat.std)

    rows: List[Dict[str, object]] = []
    for claim in claim_list:
        sigma = by_task.get(claim.benchmark)
        if sigma is None:
            rows.append(
                {
                    "claim_id": claim.claim_id,
                    "paper": claim.paper,
                    "benchmark": claim.benchmark,
                    "gap": round(claim.abs_gap, 2),
                    "sigma_seed": None,
                    "ses": None,
                    "verdict": "benchmark not replicated yet",
                    "reports_variance": claim.reports_variance,
                }
            )
            continue
        scored = seed_equivalent_significance(claim.abs_gap, sigma, floor, strong)
        rows.append(
            {
                "claim_id": claim.claim_id,
                "paper": claim.paper,
                "benchmark": claim.benchmark,
                "gap": round(claim.abs_gap, 2),
                "sigma_seed": round(sigma, 3),
                "ses": round(float(scored["ses"]), 2),
                "verdict": scored["verdict"],
                "reports_variance": claim.reports_variance,
            }
        )
    return rows


def headline(audit_rows: Sequence[Dict[str, object]], floor: float = 1.0) -> str:
    """The one sentence that goes in the abstract."""
    scored = [r for r in audit_rows if r.get("ses") is not None]
    if not scored:
        return "no claims could be scored yet"
    below = sum(1 for r in scored if float(r["ses"]) < floor)
    return (
        f"{below} of {len(scored)} harvested gaps "
        f"({100.0 * below / len(scored):.0f}%) fall below one seed "
        "standard deviation"
    )


def to_markdown(rows: Sequence[Dict[str, object]]) -> str:
    """Render any list of flat dicts as a markdown table."""
    if not rows:
        return "(no rows)"
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join("" if row[h] is None else str(row[h]) for h in headers) + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise a seed sweep and audit published claims"
    )
    config_mod.add_config_args(parser)
    parser.add_argument(
        "--runs",
        default=None,
        help="run root to scan; defaults to output_dir/experiment name",
    )
    parser.add_argument(
        "--out",
        default="results",
        help="directory for the analysis artefacts",
    )
    parser.add_argument(
        "--skip-claims",
        action="store_true",
        help="summarise the sweep only, before the claim CSV is filled in",
    )
    args = parser.parse_args()

    cfg = config_mod.config_from_args(args)
    run_root = args.runs or os.path.join(
        cfg["experiment"]["output_dir"], cfg["experiment"]["name"]
    )
    payloads = load_runs(run_root)
    print(f"[analysis] loaded {len(payloads)} runs from {run_root}")

    stats = summarise_all(
        payloads,
        cfg["analysis"]["bootstrap_resamples"],
        cfg["analysis"]["confidence_level"],
    )
    os.makedirs(args.out, exist_ok=True)

    seed_rows = [s.row() for s in stats]
    print("\n== seed variance by cell ==")
    print(to_markdown(seed_rows))
    with open(os.path.join(args.out, "seed_stats.json"), "w", encoding="utf-8") as fh:
        json.dump([{**s.row(), "scores": s.scores} for s in stats], fh, indent=2)

    decompositions = []
    for model in sorted({s.model for s in stats}):
        for task in sorted({s.task for s in stats if s.model == model}):
            try:
                decompositions.append(variance_decomposition(payloads, model, task))
            except (KeyError, ValueError) as error:
                print(f"[analysis] skipped decomposition {model}/{task}: {error}")
    if decompositions:
        print("\n== seed variance vs prompt variance ==")
        print(to_markdown(decompositions))
        with open(
            os.path.join(args.out, "variance_decomposition.json"), "w", encoding="utf-8"
        ) as fh:
            json.dump(decompositions, fh, indent=2)

    if args.skip_claims:
        return

    try:
        claim_list = claims_mod.load_claims(cfg["analysis"]["claims_csv"])
    except FileNotFoundError as error:
        print(f"[analysis] {error}")
        return

    print("\n== claim database ==")
    print(claims_mod.describe(claim_list))
    audit_rows = audit_claims(
        claim_list,
        stats,
        cfg["analysis"]["ses_floor"],
        cfg["analysis"]["ses_strong"],
    )
    print("\n== claim audit ==")
    print(to_markdown(audit_rows))
    print("\n" + headline(audit_rows, cfg["analysis"]["ses_floor"]))
    with open(os.path.join(args.out, "claim_audit.json"), "w", encoding="utf-8") as fh:
        json.dump(audit_rows, fh, indent=2)
    print(f"\n[analysis] artefacts written to {args.out}/")


if __name__ == "__main__":
    main()
