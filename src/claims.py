"""The claim database: published SLM comparisons, harvested by hand.

This is the half of the project that is not GPU work, and it is the half that
makes the paper a contribution rather than a benchmark run. Each row is one
comparison that some paper presented as evidence that method or model A beats
B, recorded with the exact reported numbers and, crucially, with whether the
original authors reported any measure of variance at all.

Harvesting rules, which belong in the paper's appendix as well:
  1. One row per comparison the authors actually draw a conclusion from, not
     one row per table cell.
  2. Record numbers exactly as printed, including the number of decimals; the
     printed precision is itself evidence about implied confidence.
  3. `reports_variance` is true only if the paper gives a standard deviation,
     a confidence interval, or an explicit number of seeds. "We average over
     runs" with no count is false.
  4. Never adjust or recompute a reported number. If it looks wrong, record it
     as printed and note it in `notes`.
  5. Log the paper's arXiv id so a reader can check every row.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, fields
from typing import List, Optional

CSV_COLUMNS = [
    "claim_id",
    "paper",
    "arxiv_id",
    "year",
    "benchmark",
    "model_a",
    "model_b",
    "score_a",
    "score_b",
    "params_b_millions",
    "reports_variance",
    "n_seeds_reported",
    "notes",
]


@dataclass
class Claim:
    """One published comparison, as printed."""

    claim_id: str
    paper: str
    arxiv_id: str
    year: int
    benchmark: str
    model_a: str
    model_b: str
    score_a: float
    score_b: float
    params_b_millions: Optional[float]
    reports_variance: bool
    n_seeds_reported: Optional[int]
    notes: str = ""

    @property
    def gap(self) -> float:
        """The improvement the paper claims, in benchmark points."""
        return self.score_b - self.score_a

    @property
    def abs_gap(self) -> float:
        return abs(self.gap)


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_optional_float(value: str) -> Optional[float]:
    value = str(value).strip()
    return float(value) if value else None


def _parse_optional_int(value: str) -> Optional[int]:
    value = str(value).strip()
    return int(float(value)) if value else None


def load_claims(path: str) -> List[Claim]:
    """Read the claim CSV. Raises a readable error if a column is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no claim database at {path}; copy experiments/claims_template.csv "
            "to experiments/claims.csv and start filling it in"
        )
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(CSV_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        claims = []
        for row in reader:
            if not (row.get("claim_id") or "").strip():
                continue  # blank spacer line
            claims.append(
                Claim(
                    claim_id=row["claim_id"].strip(),
                    paper=row["paper"].strip(),
                    arxiv_id=row["arxiv_id"].strip(),
                    year=int(row["year"]),
                    benchmark=row["benchmark"].strip(),
                    model_a=row["model_a"].strip(),
                    model_b=row["model_b"].strip(),
                    score_a=float(row["score_a"]),
                    score_b=float(row["score_b"]),
                    params_b_millions=_parse_optional_float(
                        row["params_b_millions"]
                    ),
                    reports_variance=_parse_bool(row["reports_variance"]),
                    n_seeds_reported=_parse_optional_int(row["n_seeds_reported"]),
                    notes=row.get("notes", "").strip(),
                )
            )
    return claims


def write_template(path: str) -> str:
    """Write an empty claim CSV with the right header and one example row."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "claim_id": "EXAMPLE-01",
                "paper": "Replace this row with a real paper you have read",
                "arxiv_id": "0000.00000",
                "year": 2025,
                "benchmark": "arc_easy",
                "model_a": "baseline as named in the paper",
                "model_b": "proposed model as named in the paper",
                "score_a": 0.0,
                "score_b": 0.0,
                "params_b_millions": "",
                "reports_variance": "false",
                "n_seeds_reported": "",
                "notes": "delete this example row before running the analysis",
            }
        )
    return path


def describe(claims: List[Claim]) -> str:
    """Harvest summary. Print this in the paper's data section."""
    if not claims:
        return "no claims loaded"
    with_variance = sum(1 for c in claims if c.reports_variance)
    gaps = sorted(c.abs_gap for c in claims)
    median = gaps[len(gaps) // 2]
    lines = [
        f"claims: {len(claims)}",
        f"distinct papers: {len({c.paper for c in claims})}",
        f"benchmarks: {sorted({c.benchmark for c in claims})}",
        f"report any variance: {with_variance} of {len(claims)} "
        f"({100.0 * with_variance / len(claims):.0f}%)",
        f"absolute gap: min={gaps[0]:.2f} median={median:.2f} max={gaps[-1]:.2f}",
    ]
    return "\n".join(lines)


CLAIM_FIELD_NAMES = [f.name for f in fields(Claim)]
