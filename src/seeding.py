"""Every source of randomness, in one file.

This module is the scientific instrument of the project. If seeding is sloppy
or partially applied, the variance you measure is not the variance you claim
to measure, and the whole paper collapses. So it is deliberately boring and
deliberately explicit.

What a single integer seed actually controls in a fine-tuning run:
  1. weight initialisation of any newly added parameters
  2. data order (shuffling) and any data-level sampling
  3. dropout masks during training
  4. anything else that calls into the global RNGs

What it does NOT control, and why the `deterministic_kernels` ablation exists:
  5. nondeterministic GPU kernel reductions, which add their own jitter even
     at a fixed seed. Running the same seed twice with kernels left
     nondeterministic gives you the floor beneath the floor.
"""

from __future__ import annotations

import os
import random
from typing import List

import numpy as np


def set_all_seeds(seed: int, deterministic_kernels: bool = False) -> None:
    """Seed Python, NumPy and (if installed) PyTorch from one integer."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_kernels:
        # CUBLAS_WORKSPACE_CONFIG must be set before the first CUDA context is
        # created, which is why this lives here and not in the training loop.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, RuntimeError):
            # Older torch, or an op with no deterministic implementation.
            pass


def seed_worker(worker_id: int) -> None:
    """DataLoader worker init function, so workers are seeded too."""
    try:
        import torch

        worker_seed = torch.initial_seed() % 2**32
    except ImportError:
        worker_seed = worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def default_seeds(count: int = 10, start: int = 0) -> List[int]:
    """Contiguous seeds starting at `start`.

    Contiguous and boring on purpose. Hand-picked seeds (3407 and friends) are
    exactly the selection effect this project is trying to measure, so the
    sweep must not get to choose them.
    """
    return list(range(start, start + count))


def fingerprint() -> dict:
    """Environment fingerprint stored with each run.

    Reviewers of a variance paper will ask whether the spread came from the
    seed or from a drifting environment. This answers that in one dict.
    """
    import platform
    import sys

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except ImportError:
        info["torch"] = None
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        info["transformers"] = None
    return info
