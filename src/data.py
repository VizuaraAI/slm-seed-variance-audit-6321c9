"""Dataset loading. Nothing large is committed; everything downloads.

Two kinds of data are needed:

* an SFT corpus, used to fine-tune the same model K times under K seeds
  (the training-variance experiment);
* multiple-choice benchmarks, used to score every resulting checkpoint
  (the evaluation-variance experiment).

Benchmark items are normalised into one tiny dataclass, `MultipleChoiceItem`,
so that `evaluate.py` does not need to know which benchmark it is scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import load_dataset

# Benchmark registry: hub id, config name, and split to score.
# Keep these at the "easy to download, cheap to score" end; the point of the
# project is many repeated runs, not one maximal evaluation.
BENCHMARKS: Dict[str, Dict[str, Optional[str]]] = {
    "arc_easy": {"path": "allenai/ai2_arc", "name": "ARC-Easy", "split": "validation"},
    "arc_challenge": {
        "path": "allenai/ai2_arc",
        "name": "ARC-Challenge",
        "split": "validation",
    },
    "hellaswag": {"path": "Rowan/hellaswag", "name": None, "split": "validation"},
    "piqa": {"path": "ybisk/piqa", "name": None, "split": "validation"},
    "sciq": {"path": "allenai/sciq", "name": None, "split": "validation"},
}


@dataclass
class MultipleChoiceItem:
    """One benchmark question, benchmark-agnostic."""

    context: str
    choices: List[str]
    answer_index: int
    task: str


def load_benchmark(
    task: str, max_examples: Optional[int] = None
) -> List[MultipleChoiceItem]:
    """Load one benchmark and normalise it into MultipleChoiceItem objects."""
    if task not in BENCHMARKS:
        raise KeyError(
            f"unknown task {task!r}; known tasks: {sorted(BENCHMARKS)}"
        )
    spec = BENCHMARKS[task]
    dataset = load_dataset(spec["path"], spec["name"], split=spec["split"])
    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    items: List[MultipleChoiceItem] = []
    for row in dataset:
        item = _normalise(task, row)
        if item is not None:
            items.append(item)
    if not items:
        raise RuntimeError(f"loaded 0 usable items for task {task!r}")
    return items


def _normalise(task: str, row: Dict[str, Any]) -> Optional[MultipleChoiceItem]:
    """Per-benchmark field mapping. Rows with a bad label are skipped."""
    if task.startswith("arc"):
        choices = list(row["choices"]["text"])
        labels = list(row["choices"]["label"])
        if row["answerKey"] not in labels:
            return None
        return MultipleChoiceItem(
            context=row["question"].strip(),
            choices=choices,
            answer_index=labels.index(row["answerKey"]),
            task=task,
        )

    if task == "hellaswag":
        # HellaSwag's label is a string digit, and its context is ctx, not a
        # question. The de-tokenisation artefacts are left untouched on
        # purpose: cleaning them would change the benchmark.
        label = str(row.get("label", "")).strip()
        if not label.isdigit():
            return None
        context = f"{row['activity_label']}: {row['ctx']}".strip()
        return MultipleChoiceItem(
            context=context,
            choices=[ending.strip() for ending in row["endings"]],
            answer_index=int(label),
            task=task,
        )

    if task == "piqa":
        if int(row["label"]) not in (0, 1):
            return None
        return MultipleChoiceItem(
            context=row["goal"].strip(),
            choices=[row["sol1"].strip(), row["sol2"].strip()],
            answer_index=int(row["label"]),
            task=task,
        )

    if task == "sciq":
        distractors = [
            row["distractor1"].strip(),
            row["distractor2"].strip(),
            row["distractor3"].strip(),
        ]
        # The correct answer is always placed last and the index recorded, so
        # option order is fixed rather than seed-dependent. Option order is a
        # separate variance source; it is studied in its own ablation, not
        # silently mixed into this one.
        choices = distractors + [row["correct_answer"].strip()]
        return MultipleChoiceItem(
            context=row["question"].strip(),
            choices=choices,
            answer_index=len(choices) - 1,
            task=task,
        )

    raise KeyError(f"no normaliser implemented for task {task!r}")


def load_sft_dataset(
    path: str, split: str = "train", max_examples: Optional[int] = None
):
    """Load the instruction-tuning corpus used for the training runs."""
    dataset = load_dataset(path, split=split)
    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    return dataset


def format_sft_example(row: Dict[str, Any]) -> str:
    """Render one SFT row as a single training string.

    Alpaca-style fields are handled first; anything else falls back to a
    `text` field so you can point `data.sft_dataset` at another corpus
    without editing code.
    """
    if "instruction" in row and "output" in row:
        instruction = row["instruction"].strip()
        context = (row.get("input") or "").strip()
        response = row["output"].strip()
        if context:
            prompt = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{context}\n\n### Response:\n"
            )
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        return prompt + response
    if "text" in row:
        return str(row["text"])
    raise KeyError(
        "SFT row has neither (instruction, output) nor text; add a formatter"
    )
