#!/usr/bin/env python3
"""Coordinate-aware PrismNet inference and engineering QC for one STAMP H5."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

CLASS_NAMES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
COLORS = np.asarray(
    [
        [255, 153, 153],
        [205, 205, 205],
        [153, 76, 0],
        [0, 80, 255],
        [255, 255, 153],
        [255, 0, 0],
        [0, 200, 0],
        [255, 153, 51],
        [128, 0, 128],
    ],
    dtype=np.uint8,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--feature-h5", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--wsi", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--target-microns", type=float, default=256.0)
    p.add_argument("--patch-size", type=int, default=224)
    p.add_argument("--source-wsi-mpp", type=float, required=True)
    p.add_argument("--sample-id", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    required_outputs = [
        a.output_dir / "tile_predictions.tsv",
        a.output_dir / "coordinate_probmap.npz",
        a.output_dir / "official_centered_probmap.npy",
        a.output_dir / "semantic_overlay.png",
        a.output_dir / "qc_tile_montage.png",
        a.output_dir / "qc_summary.json",
    ]
    if not a.overwrite and any(p.exists() for p in required_outputs):
        raise SystemExit("Output exists; pass --overwrite to replace the complete QC set")

    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    with h5py.File(a.feature_h5, "r") as f:
        feats = torch.from_numpy(f["feats"][:]).float()
        coords = np.asarray(f["coords"][:], dtype=np.int64)
        h5_attrs = {k: str(v) for k, v in f.attrs.items()}
    if feats.ndim != 2 or feats.shape[1] != 1024:
        raise ValueError(f"Expected Nx1024 UNI features, got {tuple(feats.shape)}")
    if coords.shape != (len(feats), 2):
        raise ValueError(f"Coordinate shape mismatch: {coords.shape} for {len(feats)} features")
    if len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError("Duplicate STAMP coordinates detected")

    pca_mu = ckpt["pca_mu"].float()
    pca_w = ckpt["pca_W"].float()
    weight = ckpt["state_dict"]["weight"].float()
    bias = ckpt["state_dict"]["bias"].float()
    probs = F.softmax(((feats - pca_mu) @ pca_w) @ weight.T + bias, dim=1).numpy()
    if not np.allclose(probs.sum(1), 1.0, atol=1e-5):
        raise ValueError("PrismNet probabilities do not sum to one")

    labels = probs.argmax(1)
    confidence = probs.max(1)
    target_mpp = a.target_microns / a.patch_size
    centers_x_um = (coords[:, 0] + a.patch_size / 2) * target_mpp
    centers_y_um = (coords[:, 1] + a.patch_size / 2) * target_mpp
    table = pd.DataFrame(
        {
            "sample_id": a.sample_id,
            "tile_id": np.arange(len(coords)),
            "x_target_px": coords[:, 0],
            "y_target_px": coords[:, 1],
            "center_x_um": centers_x_um,
            "center_y_um": centers_y_um,
            "predicted_class": [CLASS_NAMES[i] for i in labels],
            "confidence": confidence,
        }
    )
    for i, name in enumerate(CLASS_NAMES):
        table[f"p_{name}"] = probs[:, i]
    table.to_csv(a.output_dir / "tile_predictions.tsv", sep="\t", index=False)

    gx = coords[:, 0] // a.patch_size
    gy = coords[:, 1] // a.patch_size
    width = int(gx.max()) + 1
    height = int(gy.max()) + 1
    coord_probs = np.full((height, width, len(CLASS_NAMES)), np.nan, dtype=np.float32)
    coord_probs[gy, gx] = probs.astype(np.float32)
    np.savez_compressed(
        a.output_dir / "coordinate_probmap.npz",
        probabilities=coord_probs,
        valid=np.isfinite(coord_probs[..., 0]),
        class_names=np.asarray(CLASS_NAMES),
        tile_edge_um=np.asarray(a.target_microns),
        target_mpp=np.asarray(target_mpp),
    )

    official_height = int((coords[:, 0] / a.patch_size).max())
    official_width = int((coords[:, 1] / a.patch_size).max())
    matrix_size = max(official_height, official_width) + 16
    ox = (matrix_size - official_height) // 2
    oy = (matrix_size - official_width) // 2
    official = np.zeros((matrix_size, matrix_size, len(CLASS_NAMES)), dtype=np.float64)
    official[gx + ox, gy + oy] = probs
    np.save(a.output_dir / "official_centered_probmap.npy", official)

    semantic = np.zeros((height, width, 3), dtype=np.uint8)
    valid = np.isfinite(coord_probs[..., 0])
    semantic[valid] = COLORS[
        np.nanargmax(np.where(np.isnan(coord_probs), -1, coord_probs), axis=2)[valid]
    ]
    Image.fromarray(semantic).resize((width * 4, height * 4), Image.Resampling.NEAREST).save(
        a.output_dir / "semantic_map.png"
    )

    import openslide

    slide = openslide.OpenSlide(str(a.wsi))
    thumb = slide.get_thumbnail((1600, 1600)).convert("RGB")
    overlay = Image.new("RGBA", thumb.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    scale_x = thumb.width / slide.dimensions[0]
    scale_y = thumb.height / slide.dimensions[1]
    target_to_level0 = target_mpp / a.source_wsi_mpp
    for (x, y), lab, conf in zip(coords, labels, confidence):
        x0 = x * target_to_level0 * scale_x
        y0 = y * target_to_level0 * scale_y
        x1 = (x + a.patch_size) * target_to_level0 * scale_x
        y1 = (y + a.patch_size) * target_to_level0 * scale_y
        color = tuple(int(v) for v in COLORS[lab])
        draw.rectangle((x0, y0, x1, y1), fill=(*color, int(45 + 125 * conf)))
    Image.alpha_composite(thumb.convert("RGBA"), overlay).convert("RGB").save(
        a.output_dir / "semantic_overlay.png"
    )

    rng = np.random.default_rng(20260728)
    chosen = []
    per_class = max(1, 100 // len(CLASS_NAMES))
    for c in range(len(CLASS_NAMES)):
        idx = np.flatnonzero(labels == c)
        if len(idx):
            order = idx[np.argsort(-confidence[idx])]
            top = order[: max(1, per_class // 2)]
            rest = np.setdiff1d(idx, top, assume_unique=False)
            random_part = (
                rng.choice(rest, min(per_class - len(top), len(rest)), replace=False)
                if len(rest)
                else []
            )
            chosen.extend(np.asarray(top).tolist() + np.asarray(random_part).tolist())
    if len(chosen) < min(100, len(coords)):
        remaining = np.setdiff1d(np.arange(len(coords)), np.asarray(chosen, dtype=int))
        chosen.extend(
            rng.choice(remaining, min(100 - len(chosen), len(remaining)), replace=False).tolist()
        )
    chosen = chosen[:100]
    tile_level0 = int(round(a.target_microns / a.source_wsi_mpp))
    cell = 128
    montage = Image.new("RGB", (10 * cell, 10 * (cell + 18)), "white")
    md = ImageDraw.Draw(montage)
    for pos, idx in enumerate(chosen):
        x0 = int(round(coords[idx, 0] * target_to_level0))
        y0 = int(round(coords[idx, 1] * target_to_level0))
        tile = slide.read_region((x0, y0), 0, (tile_level0, tile_level0)).convert("RGB")
        tile = tile.resize((cell, cell), Image.Resampling.BILINEAR)
        col, row = pos % 10, pos // 10
        montage.paste(tile, (col * cell, row * (cell + 18)))
        md.text(
            (col * cell + 2, row * (cell + 18) + cell),
            f"{idx}:{CLASS_NAMES[labels[idx]]} {confidence[idx]:.2f}",
            fill="black",
        )
    montage.save(a.output_dir / "qc_tile_montage.png")
    slide.close()

    non_back = labels != CLASS_NAMES.index("BACK")
    counts = {name: int(np.sum(labels == i)) for i, name in enumerate(CLASS_NAMES)}
    fractions = {
        name: (float(np.sum(labels[non_back] == i) / non_back.sum()) if non_back.sum() else None)
        for i, name in enumerate(CLASS_NAMES)
        if name != "BACK"
    }
    qc = {
        "sample_id": a.sample_id,
        "n_tiles": int(len(coords)),
        "feature_shape": list(feats.shape),
        "coord_min": coords.min(0).tolist(),
        "coord_max": coords.max(0).tolist(),
        "coordinate_grid_shape_yx": [height, width],
        "target_microns": a.target_microns,
        "patch_size": a.patch_size,
        "target_mpp": target_mpp,
        "source_wsi_mpp": a.source_wsi_mpp,
        "class_counts": counts,
        "non_background_fractions": fractions,
        "confidence_quantiles": {
            str(q): float(np.quantile(confidence, q)) for q in [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
        },
        "montage_tile_ids": chosen,
        "h5_attrs": h5_attrs,
        "hashes": {
            "feature_h5": sha256(a.feature_h5),
            "checkpoint": sha256(a.checkpoint),
            "wsi": sha256(a.wsi),
        },
        "interpretation": "Engineering QC only; predicted semantic classes are not pathology ground truth.",
    }
    (a.output_dir / "qc_summary.json").write_text(json.dumps(qc, indent=2) + "\n")
    print(
        json.dumps({"status": "success", "n_tiles": len(coords), "output_dir": str(a.output_dir)})
    )


if __name__ == "__main__":
    main()
