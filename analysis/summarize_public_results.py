#!/usr/bin/env python3
"""Recompute the public slide-level descriptive summary without private inputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slide-effects",
        type=Path,
        default=Path("results/pathprism_slide_effects.tsv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/aggregate_summary.json"),
    )
    args = parser.parse_args()

    with args.slide_effects.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    values = [float(row["mean_paired_delta_lym_fraction_112um"]) for row in rows]
    if not values:
        raise SystemExit("No slide-level values found")

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    expected_n = summary["pathprism_cross_model_consistency"]["detected_slides_total"]
    expected_mean = summary["pathprism_cross_model_consistency"][
        "mean_of_slide_mean_paired_deltas"
    ]
    observed_mean = sum(values) / len(values)

    if len(values) != expected_n:
        raise SystemExit(f"Slide count mismatch: {len(values)} != {expected_n}")
    if sum(value > 0 for value in values) != expected_n:
        raise SystemExit("Not every retained detected slide has a positive paired effect")
    if abs(observed_mean - expected_mean) > 0.0001:
        raise SystemExit(f"Mean mismatch: {observed_mean:.6f} != {expected_mean:.6f}")

    print(
        json.dumps(
            {
                "status": "pass",
                "n_slides": len(values),
                "positive_direction": sum(value > 0 for value in values),
                "mean_paired_delta": round(observed_mean, 6),
                "claim": "cross-model spatial consistency only",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
