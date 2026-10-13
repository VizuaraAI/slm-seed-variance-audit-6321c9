"""Config loading.

The config is a plain dict so you can print it, diff it, and dump it next to
every run. Dumping the resolved config into each run directory matters more
than usual here: the claim of this project is that two runs were identical
except for the seed, and the only way to defend that claim is to have the
resolved config on disk for both.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Any, Dict

import yaml

DEFAULT_CONFIG_PATH = os.path.join("configs", "default.yaml")


def load_config(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Read a YAML config file into a dict."""
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"config at {path} did not parse into a mapping")
    config["_config_path"] = path
    return config


def apply_overrides(config: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    """Apply `section.key=value` strings onto a config.

    Values are parsed as JSON when possible, so `train.epochs=2` becomes an
    int and `eval.tasks=["arc_easy"]` becomes a list.
    """
    config = copy.deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override {override!r} is not of the form key=value")
        dotted_key, raw_value = override.split("=", 1)
        try:
            value: Any = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        target = config
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return config


def run_dir(config: Dict[str, Any], model_name: str, seed: int) -> str:
    """Deterministic run directory: one directory per (model, seed)."""
    safe_model = model_name.replace("/", "__")
    return os.path.join(
        config["experiment"]["output_dir"],
        config["experiment"]["name"],
        safe_model,
        f"seed{seed:03d}",
    )


def save_resolved_config(config: Dict[str, Any], directory: str) -> str:
    """Write the exact config used by a run into that run's directory."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "resolved_config.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
    return path


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Shared CLI flags for every entry point in this repo."""
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="path to the YAML config",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set train.epochs=2",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return apply_overrides(load_config(args.config), args.overrides)
