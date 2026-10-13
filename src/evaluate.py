"""Scoring: length-normalised log-likelihood over multiple-choice benchmarks.

Why implement scoring here instead of only calling lm-evaluation-harness?
Because this project needs to vary the prompt surface form while holding
everything else fixed, and it needs the per-item correctness vector (not just
the headline accuracy) so that `analysis.py` can bootstrap a confidence
interval. Both are easier with 150 lines you fully control. The harness is
still used as an independent cross-check (see scripts/run_harness_eval.sh);
if the two disagree, that disagreement is itself a result worth reporting.

Scoring rule: for each candidate continuation, sum the token log-probabilities
of the continuation given the prompt, then divide by the continuation's token
count. Length normalisation matters for HellaSwag-style tasks where endings
differ a lot in length. The paper reports both normalised and unnormalised
accuracy, because the choice of normalisation is yet another knob that can
move a number by more than a claimed gap.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

from . import config as config_mod
from . import data as data_mod
from . import modeling, seeding

# Three surface forms of the same question. Nothing about the underlying task
# changes; only the wrapper text does.
PROMPT_TEMPLATES: Dict[str, str] = {
    "plain": "{context}",
    "qa": "Question: {context}\nAnswer:",
    "instruct": (
        "### Instruction:\nAnswer the following question.\n\n"
        "### Input:\n{context}\n\n### Response:"
    ),
}


@dataclass
class TaskResult:
    """Scores for one (task, prompt template) pair."""

    task: str
    template: str
    n_items: int
    accuracy: float
    accuracy_unnormalised: float
    correct: List[int]

    def summary(self) -> Dict[str, object]:
        row = asdict(self)
        row.pop("correct")
        return row


def build_prompt(template: str, context: str) -> str:
    if template not in PROMPT_TEMPLATES:
        raise KeyError(
            f"unknown template {template!r}; known: {sorted(PROMPT_TEMPLATES)}"
        )
    return PROMPT_TEMPLATES[template].format(context=context.strip())


def _continuation_logprob(model, tokenizer, prompt: str, continuation: str, device):
    """Total and per-token log-probability of `continuation` after `prompt`."""
    import torch

    # A leading space makes the continuation tokenise the way it would inside
    # running text, which is what the model saw during pretraining.
    if not continuation.startswith((" ", "\n")):
        continuation = " " + continuation

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    cont_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]
    if not cont_ids:
        return float("-inf"), float("-inf")

    input_ids = torch.tensor([prompt_ids + cont_ids], device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits.float()

    # Predict position i from position i-1, so the logits that score the
    # continuation start one step before it begins.
    log_probs = torch.log_softmax(logits[0, :-1, :], dim=-1)
    targets = input_ids[0, 1:]
    start = len(prompt_ids) - 1
    cont_log_probs = log_probs[start:, :].gather(
        1, targets[start:].unsqueeze(-1)
    ).squeeze(-1)

    total = float(cont_log_probs.sum().item())
    return total, total / len(cont_ids)


def score_task(
    model,
    tokenizer,
    items: Sequence[data_mod.MultipleChoiceItem],
    template: str,
    device,
) -> TaskResult:
    """Score every item in one task under one prompt template."""
    correct_norm: List[int] = []
    correct_raw: List[int] = []

    for item in items:
        prompt = build_prompt(template, item.context)
        totals, per_token = [], []
        for choice in item.choices:
            total, mean = _continuation_logprob(
                model, tokenizer, prompt, choice, device
            )
            totals.append(total)
            per_token.append(mean)
        pred_norm = max(range(len(per_token)), key=per_token.__getitem__)
        pred_raw = max(range(len(totals)), key=totals.__getitem__)
        correct_norm.append(int(pred_norm == item.answer_index))
        correct_raw.append(int(pred_raw == item.answer_index))

    n = len(correct_norm)
    return TaskResult(
        task=items[0].task,
        template=template,
        n_items=n,
        accuracy=100.0 * sum(correct_norm) / n,
        accuracy_unnormalised=100.0 * sum(correct_raw) / n,
        correct=correct_norm,
    )


def evaluate_checkpoint(cfg: Dict, model_path: str, seed: int) -> Dict[str, object]:
    """Evaluate one checkpoint over every (task, template) pair."""
    # Evaluation is deterministic given the checkpoint, but the seed is still
    # set so that any accidental stochasticity (dropout left on, sampling)
    # cannot leak in unnoticed.
    seeding.set_all_seeds(seed)
    device = modeling.pick_device()
    model, tokenizer = modeling.load_model_and_tokenizer(
        model_path,
        dtype=cfg["model"]["dtype"],
        trust_remote_code=cfg["model"]["trust_remote_code"],
        device=device,
    )
    model.eval()

    results: List[Dict[str, object]] = []
    for task in cfg["eval"]["tasks"]:
        items = data_mod.load_benchmark(task, cfg["eval"]["max_eval_examples"])
        for template in cfg["eval"]["prompt_templates"]:
            task_result = score_task(model, tokenizer, items, template, device)
            print(
                f"[eval] seed={seed} {task}/{template}: "
                f"acc={task_result.accuracy:.2f} (n={task_result.n_items})"
            )
            results.append(
                {**task_result.summary(), "correct": task_result.correct}
            )

    return {
        "model": model_path,
        "base_model": cfg["model"]["name"],
        "seed": seed,
        "results": results,
        "environment": seeding.fingerprint(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score one checkpoint across benchmarks and prompt templates"
    )
    config_mod.add_config_args(parser)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--model-path",
        default=None,
        help="checkpoint directory; defaults to the base model (zero-shot)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="where to write eval.json; defaults to the run directory",
    )
    args = parser.parse_args()

    cfg = config_mod.config_from_args(args)
    directory = config_mod.run_dir(cfg, cfg["model"]["name"], args.seed)
    model_path = args.model_path or cfg["model"]["name"]
    payload = evaluate_checkpoint(cfg, model_path, args.seed)

    out_path = args.out or os.path.join(directory, "eval.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
