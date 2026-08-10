#!/usr/bin/env python3
"""Aggregate locked public STAD validation outputs without tile pseudoreplication."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--pan-root", type=Path, required=True)
    p.add_argument("--pathprism-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        a.output_dir / "cohort_slide_summary.tsv",
        a.output_dir / "cohort_object_paired.tsv",
        a.output_dir / "cohort_summary.json",
    ]
    if not a.overwrite and any(x.exists() for x in outputs):
        raise SystemExit("Output exists; use --overwrite")
    selected = list(csv.DictReader(a.manifest.open(), delimiter="\t"))
    slides, objects = [], []
    for row in selected:
        sample = row["Slide_ID"]
        pan = read_json(a.pan_root / sample / "summary.json")
        slide_record = {
            "sample_id": sample,
            "selection_rank": int(row["selection_rank"]),
            "pan_status": pan.get("pipeline_status") if pan else "missing",
            "total_tls": pan.get("total_objects") if pan else np.nan,
            "stable_tls": pan.get("stable_count") if pan else np.nan,
            "E_TLS": pan.get("consensus_counts", {}).get("E-TLS") if pan else np.nan,
            "P_TLS": pan.get("consensus_counts", {}).get("P-TLS") if pan else np.nan,
            "S_TLS": pan.get("consensus_counts", {}).get("S-TLS") if pan else np.nan,
            "uncertain_TLS": pan.get("consensus_counts", {}).get("uncertain") if pan else np.nan,
        }
        for scale in [112, 256]:
            run = a.pathprism_root / sample / f"scale_{scale}um"
            qc = read_json(run / "prismnet/qc_summary.json")
            pilot = read_json(run / "tls_tissue_graph/pilot_summary.json")
            prefix = f"s{scale}_"
            slide_record[prefix + "status"] = pilot.get("status") if pilot else "missing"
            slide_record[prefix + "n_tiles"] = qc.get("n_tiles") if qc else np.nan
            slide_record[prefix + "median_confidence"] = (
                qc.get("confidence_quantiles", {}).get("0.5") if qc else np.nan
            )
            inner = pilot.get("primary_inner_250um_lym", {}) if pilot else {}
            slide_record[prefix + "tls_inner_positive"] = inner.get("tls_positive_count", np.nan)
            slide_record[prefix + "tls_inner_total"] = inner.get("tls_total", np.nan)
            paired_path = run / "tls_tissue_graph/primary_paired_differences.tsv"
            if paired_path.exists():
                df = pd.read_csv(paired_path, sep="\t")
                if len(df):
                    df.insert(0, "sample_id", sample)
                    df.insert(1, "scale_um", scale)
                    objects.append(df)
                    for feature in [
                        "ring_0_250_fraction_LYM_paired_difference",
                        "mean_probability_LYM_paired_difference",
                    ]:
                        slide_record[prefix + "mean_" + feature] = float(df[feature].mean())
                else:
                    slide_record[prefix + "mean_ring_0_250_fraction_LYM_paired_difference"] = np.nan
                    slide_record[prefix + "mean_mean_probability_LYM_paired_difference"] = np.nan
        slides.append(slide_record)
    slide_df = pd.DataFrame(slides).sort_values("selection_rank")
    object_df = pd.concat(objects, ignore_index=True, sort=False) if objects else pd.DataFrame()
    slide_df.to_csv(outputs[0], sep="\t", index=False)
    object_df.to_csv(outputs[1], sep="\t", index=False)

    complete112 = slide_df[slide_df.s112_status == "success"]
    effect_col = "s112_mean_ring_0_250_fraction_LYM_paired_difference"
    effects = (
        complete112[effect_col].dropna().to_numpy(float)
        if effect_col in complete112
        else np.array([])
    )
    bootstrap = {}
    if len(effects) >= 2:
        rng = np.random.default_rng(20260728)
        means = np.mean(rng.choice(effects, size=(10000, len(effects)), replace=True), axis=1)
        bootstrap = {
            "slide_mean_effect": float(effects.mean()),
            "cluster_bootstrap_95ci": [
                float(np.quantile(means, 0.025)),
                float(np.quantile(means, 0.975)),
            ],
            "positive_direction_slides": int((effects > 0).sum()),
            "nonzero_effect_slides": int(len(effects)),
        }
    summary = {
        "status": "complete" if len(complete112) == len(selected) else "partial",
        "interpretation": "locked public validation; model consistency, not pathological accuracy",
        "n_locked_slides": len(selected),
        "n_completed_primary_scale": int(len(complete112)),
        "n_zero_detection_slides": int((slide_df.total_tls.fillna(-1) == 0).sum()),
        "n_tls_objects_primary_scale": int(len(object_df[object_df.scale_um == 112]))
        if len(object_df)
        else 0,
        "primary_effect": "112-um inner 0-250 LYM fraction minus mean of 3 within-slide controls",
        "primary_slide_level_summary": bootstrap,
    }
    outputs[2].write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
