# Reproducibility guide

## What is reproducible from this repository

- dependency-free audit metric tests;
- synthetic coordinate, boundary and matched-control checks after installing
  the scientific dependencies;
- validation of the eight anonymized slide-level PathPrism effects;
- inspection of all formulas, thresholds and aggregate figures;
- reuse of the adapters with independently obtained data and checkpoints.

## What is intentionally absent

- WSI files and image patches;
- patient or clinical tables;
- object-level model predictions and review images;
- HookNet, YOLO, PrismNet, UNI or SEAL weights;
- UNI-derived feature matrices;
- local paths, cluster job IDs, logs, tokens and credentials.

## Environment

Python 3.10 or newer is recommended. OpenSlide must be installed at the system
level for WSI access. A CUDA-capable environment is recommended for the model
audit but is not needed for aggregate summaries.

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e ".[wsi,models,dev]"

Run the lightweight checks:

    python3 -m unittest discover -s tests -p "test_*.py" -v
    python3 analysis/summarize_public_results.py

Run the optional scientific-stack checks:

    python3 tests/synthetic_spatial_checks.py

## External assets

1. Download the listed TCGA-STAD slides directly from GDC.
2. Obtain HookNet-TLS, the maturity weights, PathPrism/PrismNet and UNI from
   their official sources.
3. Verify every available GDC MD5 before processing.
4. Record model revisions and hashes locally, but do not commit gated weights
   or absolute paths.
5. Replace the placeholders in configs/example.yaml locally.
6. Build an object inventory following the schema in data/README.md and keep it
   outside version control.

## Expected outputs

The full audit writes variant-level probability tables, attribution metrics,
spatial features and run manifests. These outputs may contain slide identifiers
or local paths and should remain outside the public repository unless
separately sanitized.

## Numerical scope

The aggregate summary is checked against an anonymized eight-row slide table.
The saved exploratory bootstrap interval comes from the original slide-aware
analysis and is reported for transparency; it is not recomputed from
object-level data because those records are intentionally excluded.
