"""Fine-tune one model with one seed.

Deliberately a plain PyTorch loop rather than `Trainer`. For this project the
training loop is the measuring device, so every source of randomness in it has
to be visible on one screen: the shuffle, the dropout, the init. A framework
that hides those behind a config flag makes the central claim harder to defend.

One process trains exactly one seed. The sweep is a shell loop over seeds
(scripts/run_seed_sweep.sh), which keeps the runs independent: a crash at seed
7 costs you seed 7 and nothing else.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any, Dict, List

from . import config as config_mod
from . import data as data_mod
from . import modeling, seeding


def build_batches(
    texts: List[str], tokenizer, max_length: int, batch_size: int, shuffle: bool, rng
) -> List[Dict[str, Any]]:
    """Tokenise and batch the SFT corpus.

    Batching happens once, up front, so that the data order for a given seed is
    recorded and reproducible rather than regenerated lazily.
    """
    indices = list(range(len(texts)))
    if shuffle:
        rng.shuffle(indices)

    batches = []
    for start in range(0, len(indices), batch_size):
        chunk = [texts[i] for i in indices[start : start + batch_size]]
        encoded = tokenizer(
            chunk,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        labels = encoded["input_ids"].clone()
        # Padding must not contribute to the loss, or short examples would
        # quietly reward the model for predicting pad tokens.
        labels[encoded["attention_mask"] == 0] = -100
        encoded["labels"] = labels
        batches.append(encoded)
    return batches


def train_one_seed(cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Run a single fine-tuning job and save the checkpoint."""
    import random

    import torch

    seeding.set_all_seeds(seed, cfg["train"]["deterministic_kernels"])
    device = modeling.pick_device()
    directory = config_mod.run_dir(cfg, cfg["model"]["name"], seed)
    os.makedirs(directory, exist_ok=True)
    config_mod.save_resolved_config({**cfg, "_seed": seed}, directory)

    print(f"[train] seed={seed} model={cfg['model']['name']} device={device}")
    model, tokenizer = modeling.load_model_and_tokenizer(
        cfg["model"]["name"],
        dtype=cfg["model"]["dtype"],
        trust_remote_code=cfg["model"]["trust_remote_code"],
        device=device,
    )
    n_params = modeling.count_parameters(model)
    print(f"[train] parameters: {n_params / 1e6:.1f}M")

    dataset = data_mod.load_sft_dataset(
        cfg["data"]["sft_dataset"],
        cfg["data"]["sft_split"],
        cfg["data"]["sft_max_examples"],
    )
    texts = [data_mod.format_sft_example(row) for row in dataset]

    # A seed-local RNG for data order, so the shuffle is reproducible from the
    # seed alone and independent of how many other random calls happened first.
    rng = random.Random(seed)
    batches = build_batches(
        texts,
        tokenizer,
        cfg["data"]["max_seq_length"],
        cfg["train"]["batch_size"],
        cfg["train"]["shuffle_data"],
        rng,
    )

    accum = cfg["train"]["grad_accum_steps"]
    total_steps = max(1, math.ceil(len(batches) * cfg["train"]["epochs"] / accum))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    warmup_steps = int(cfg["train"]["warmup_ratio"] * total_steps)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    history: List[Dict[str, float]] = []
    started = time.time()
    step = 0

    for epoch in range(cfg["train"]["epochs"]):
        for batch_index, batch in enumerate(batches):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / accum
            loss.backward()

            if (batch_index + 1) % accum == 0 or batch_index + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["train"]["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % cfg["train"]["log_every"] == 0:
                    entry = {
                        "step": step,
                        "epoch": epoch,
                        "loss": float(loss.item() * accum),
                        "lr": float(scheduler.get_last_lr()[0]),
                    }
                    history.append(entry)
                    print(
                        f"[train] seed={seed} step={step}/{total_steps} "
                        f"loss={entry['loss']:.4f}"
                    )

    checkpoint_dir = os.path.join(directory, "checkpoint")
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    record = {
        "seed": seed,
        "model": cfg["model"]["name"],
        "parameters": n_params,
        "steps": step,
        "wall_clock_seconds": round(time.time() - started, 1),
        "final_loss": history[-1]["loss"] if history else None,
        "history": history,
        "checkpoint": checkpoint_dir,
        "environment": seeding.fingerprint(),
    }
    with open(os.path.join(directory, "train.json"), "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    print(f"[train] seed={seed} done in {record['wall_clock_seconds']}s")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune one model, one seed")
    config_mod.add_config_args(parser)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    train_one_seed(config_mod.config_from_args(args), args.seed)


if __name__ == "__main__":
    main()
