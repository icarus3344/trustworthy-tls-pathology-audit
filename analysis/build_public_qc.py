#!/usr/bin/env python3
"""Build auditable visual QC, object anomaly tables, and mentor figures.

This script performs engineering/visual plausibility QC only. It does not
create pathological ground-truth labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    import openslide
except ImportError as exc:  # pragma: no cover - checked on the remote environment
    raise SystemExit(f"openslide-python is required: {exc}")


CLASS_COLORS = {
    "ADI": "#F2CF5B",
    "BACK": "#D9D9D9",
    "DEB": "#9C755F",
    "LYM": "#4C78A8",
    "MUC": "#B279A2",
    "MUS": "#FF9DA6",
    "NORM": "#59A14F",
    "STR": "#76B7B2",
    "TUM": "#E15759",
}
MATURITY_COLORS = {
    "E-TLS": "#F2CF5B",
    "P-TLS": "#4C78A8",
    "S-TLS": "#E15759",
    "uncertain": "#BAB0AC",
}
SLIDE_COLORS = ["#4C78A8", "#F28E2B", "#59A14F", "#E15759", "#B279A2"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pp-root", type=Path, required=True)
    p.add_argument("--pc-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--pp-analysis-name", default="public_stad_validation_v1")
    p.add_argument("--pp-cohort-name", default="public_stad_validation_v1_cohort")
    p.add_argument("--pc-analysis-name", default="public_stad_validation_v1")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def polygon_area(points: list[list[float]], mpp: float = 0.5) -> float:
    a = np.asarray(points, dtype=float)
    if len(a) < 3:
        return 0.0
    x, y = a[:, 0], a[:, 1]
    px2 = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return float(px2 * mpp * mpp)


def polygon_centroid(points: list[list[float]], mpp: float = 0.5) -> tuple[float, float]:
    a = np.asarray(points, dtype=float)
    if len(a) < 3:
        return float(a[:, 0].mean() * mpp), float(a[:, 1].mean() * mpp)
    x, y = a[:, 0], a[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    denom = cross.sum()
    if abs(denom) < 1e-12:
        return float(x.mean() * mpp), float(y.mean() * mpp)
    cx = ((x + np.roll(x, -1)) * cross).sum() / (3 * denom)
    cy = ((y + np.roll(y, -1)) * cross).sum() / (3 * denom)
    return float(cx * mpp), float(cy * mpp)


def image_texture(path: Path) -> tuple[float, float, float]:
    im = Image.open(path).convert("RGB")
    im.thumbnail((512, 512))
    a = np.asarray(im, dtype=float)
    gray = a.mean(axis=2)
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    gradient_rms = float(math.sqrt((np.mean(gx * gx) + np.mean(gy * gy)) / 2))
    tissue_fraction = float(np.mean(np.min(a, axis=2) < 235))
    white_fraction = float(np.mean(np.min(a, axis=2) > 245))
    return gradient_rms, tissue_fraction, white_fraction


def max_joint_conf(row: pd.Series, prefix: str) -> float:
    return float(max(row[f"{prefix}_p_E"], row[f"{prefix}_p_P"], row[f"{prefix}_p_S"]))


def robust_high_outlier(values: pd.Series) -> float:
    vals = values.dropna().astype(float)
    if len(vals) < 4:
        return float("inf")
    q1, q3 = vals.quantile([0.25, 0.75])
    return float(q3 + 1.5 * (q3 - q1))


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    im = Image.open(path).convert("RGB")
    im.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    return canvas


def overview_panel(sample: str, pc_dir: Path, pp_dir: Path, out: Path, stats: pd.Series) -> None:
    candidates = [
        (pc_dir / "crops/overlay_bboxes.png", "HookNet TLS boxes"),
        (pp_dir / "scale_112um/prismnet/semantic_overlay.png", "PathPrism 112 µm"),
        (pp_dir / "scale_112um/tls_tissue_graph/tls_control_overlay.png", "TLS + matched controls"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
    for ax, (path, title) in zip(axes, candidates):
        if path.exists():
            ax.imshow(Image.open(path))
        else:
            ax.text(0.5, 0.5, "not generated", ha="center", va="center")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(
        f"{sample} | TLS={int(stats.total_tls)} | stable={int(stats.stable_tls)} "
        f"| E/P/S/U={int(stats.E_TLS)}/{int(stats.P_TLS)}/{int(stats.S_TLS)}/{int(stats.uncertain_TLS)}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def read_local_wsi(
    slide: openslide.OpenSlide,
    center_um: tuple[float, float],
    half_um: float = 1000.0,
    target_mpp: float = 2.0,
) -> tuple[Image.Image, float, float, float]:
    mpp = float(slide.properties.get("openslide.mpp-x", "0.5"))
    desired_downsample = target_mpp / mpp
    level = min(range(slide.level_count), key=lambda i: abs(math.log(slide.level_downsamples[i] / desired_downsample)))
    ds = float(slide.level_downsamples[level])
    actual_mpp = mpp * ds
    x0_um, y0_um = center_um[0] - half_um, center_um[1] - half_um
    x0_px = int(math.floor(x0_um / mpp))
    y0_px = int(math.floor(y0_um / mpp))
    size = max(1, int(round(2 * half_um / actual_mpp)))
    im = slide.read_region((x0_px, y0_px), level, (size, size)).convert("RGB")
    return im, x0_um, y0_um, actual_mpp


def object_panel(
    sample: str,
    row: pd.Series,
    metadata: dict[tuple[int, float], dict],
    pc_dir: Path,
    tile112: pd.DataFrame,
    wsi: openslide.OpenSlide,
    out: Path,
) -> None:
    oid = int(row.object_id)
    pred = row["_prediction"]
    points = json.loads(pred.polygon_coordinates)
    cx, cy = polygon_centroid(points)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, pad in zip(axes[:2], [0.1, 0.25]):
        meta = metadata[(oid, pad)]
        patch = pc_dir / "crops" / f"extracted_patches_{int(pad * 100):02d}" / f"patch_{oid}.jpeg"
        im = Image.open(patch).convert("RGB")
        ax.imshow(im)
        origin = meta["crop_origin"]
        local = np.asarray(points, dtype=float) - np.asarray(origin, dtype=float)
        ax.add_patch(Polygon(local, fill=False, edgecolor="#E15759", linewidth=2))
        ax.set_title(f"{int(pad * 100)}% context: {pred[f'pad{int(pad*100):02d}_class']}")
        ax.axis("off")

    local_im, x0_um, y0_um, actual_mpp = read_local_wsi(wsi, (cx, cy))
    ax = axes[2]
    ax.imshow(local_im)
    local_tiles = tile112[
        (tile112.center_x_um >= x0_um)
        & (tile112.center_x_um <= x0_um + 2000)
        & (tile112.center_y_um >= y0_um)
        & (tile112.center_y_um <= y0_um + 2000)
        & (tile112.predicted_class != "BACK")
    ]
    tile_px = 112.0 / actual_mpp
    for t in local_tiles.itertuples():
        x = (t.center_x_um - x0_um) / actual_mpp - tile_px / 2
        y = (t.center_y_um - y0_um) / actual_mpp - tile_px / 2
        ax.add_patch(
            Rectangle(
                (x, y),
                tile_px,
                tile_px,
                facecolor=CLASS_COLORS.get(t.predicted_class, "#999999"),
                edgecolor="none",
                alpha=0.42,
            )
        )
    poly_um = np.asarray(points, dtype=float) * 0.5
    local_poly = (poly_um - np.asarray([x0_um, y0_um])) / actual_mpp
    ax.add_patch(Polygon(local_poly, fill=False, edgecolor="#E15759", linewidth=2.2))
    center_px = ((cx - x0_um) / actual_mpp, (cy - y0_um) / actual_mpp)
    for radius in [250, 500, 1000]:
        ax.add_patch(
            Circle(
                center_px,
                radius / actual_mpp,
                fill=False,
                edgecolor="#333333",
                linewidth=0.8,
                linestyle="--" if radius != 250 else "-",
                alpha=0.8,
            )
        )
    ax.set_xlim(0, local_im.width)
    ax.set_ylim(local_im.height, 0)
    ax.set_title(
        f"112 µm local map | ΔLYM250={row.s112_inner_diff:+.3f}\n"
        f"ΔpLYM={row.s112_plym_diff:+.3f}"
    )
    ax.axis("off")
    fig.suptitle(
        f"{sample} object {oid} | consensus={pred.consensus_class} | "
        f"context stable={bool(pred.context_stable)} | flags={row['flags'] or 'none'}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def make_contact_sheets(paths: list[Path], output_prefix: Path, per_page: int = 4) -> list[Path]:
    outputs = []
    thumb_size = (1000, 340)
    for page_idx in range(0, len(paths), per_page):
        batch = paths[page_idx : page_idx + per_page]
        canvas = Image.new("RGB", (1000, 340 * len(batch)), "white")
        for j, p in enumerate(batch):
            canvas.paste(fit_image(p, thumb_size), (0, 340 * j))
        out = output_prefix.parent / f"{output_prefix.name}_p{page_idx // per_page + 1}.jpg"
        canvas.save(out, quality=90)
        outputs.append(out)
    return outputs


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.05, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")


def main_result_figure(slides: pd.DataFrame, objects: pd.DataFrame, outdir: Path) -> None:
    pos = slides[slides.total_tls > 0].copy()
    samples = slides.sample_id.tolist()
    short = [s.replace("TCGA-", "") for s in samples]
    slide_color = dict(zip(samples, SLIDE_COLORS[: len(samples)]))
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    bottom = np.zeros(len(slides))
    for cls, col in MATURITY_COLORS.items():
        column = {"E-TLS": "E_TLS", "P-TLS": "P_TLS", "S-TLS": "S_TLS", "uncertain": "uncertain_TLS"}[cls]
        vals = slides[column].to_numpy(float)
        ax.bar(short, vals, bottom=bottom, color=col, label=cls)
        bottom += vals
    ax.set_ylabel("TLS candidates")
    ax.set_title("Detection and maturity consensus")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False, ncol=2)
    add_panel_label(ax, "A")

    ax = axes[0, 1]
    obj112 = objects[objects.scale_um == 112].copy()
    rng = np.random.default_rng(20260728)
    for i, sample in enumerate(pos.sample_id):
        g = obj112[obj112.sample_id == sample]
        x0, x1 = i - 0.16, i + 0.16
        for r in g.itertuples():
            jitter = rng.uniform(-0.035, 0.035)
            ax.plot(
                [x0 + jitter, x1 + jitter],
                [r.ring_0_250_fraction_LYM_control_mean, r.ring_0_250_fraction_LYM_TLS],
                color=slide_color[sample],
                alpha=0.35,
                linewidth=0.8,
            )
            ax.scatter(x0 + jitter, r.ring_0_250_fraction_LYM_control_mean, facecolor="white", edgecolor=slide_color[sample], s=22)
            ax.scatter(x1 + jitter, r.ring_0_250_fraction_LYM_TLS, color=slide_color[sample], s=22)
    ax.set_xticks(np.arange(len(pos)), [s.replace("TCGA-", "") for s in pos.sample_id], rotation=30)
    ax.set_ylabel("LYM fraction, 0–250 µm")
    ax.set_title("Object-level matched comparison (112 µm)")
    ax.text(0.02, 0.98, "open: mean of 3 controls  •  filled: TLS", transform=ax.transAxes, va="top", fontsize=9)
    add_panel_label(ax, "B")

    ax = axes[1, 0]
    x = np.arange(len(pos))
    y112 = pos.s112_mean_ring_0_250_fraction_LYM_paired_difference.to_numpy(float)
    y256 = pos.s256_mean_ring_0_250_fraction_LYM_paired_difference.to_numpy(float)
    for i in range(len(pos)):
        ax.plot([i - 0.12, i + 0.12], [y112[i], y256[i]], color="#777777", linewidth=1)
    ax.scatter(x - 0.12, y112, color="#4C78A8", s=55, label="112 µm")
    ax.scatter(x + 0.12, y256, color="#F28E2B", marker="s", s=50, label="256 µm")
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_xticks(x, [s.replace("TCGA-", "") for s in pos.sample_id], rotation=30)
    ax.set_ylabel("Slide mean paired Δ LYM fraction")
    ax.set_title("Scale sensitivity preserves direction")
    ax.legend(frameon=False)
    add_panel_label(ax, "C")

    ax = axes[1, 1]
    for i, sample in enumerate(pos.sample_id):
        g = obj112[obj112.sample_id == sample]
        vals = g.mean_probability_LYM_paired_difference.to_numpy(float)
        jitter = rng.uniform(-0.10, 0.10, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=slide_color[sample], alpha=0.75, s=28)
        ax.scatter(i, vals.mean(), color="black", marker="_", s=250, linewidth=2)
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_xticks(np.arange(len(pos)), [s.replace("TCGA-", "") for s in pos.sample_id], rotation=30)
    ax.set_ylabel("TLS − control mean p(LYM)")
    ax.set_title("Continuous p(LYM) paired effects")
    add_panel_label(ax, "D")

    fig.suptitle("Locked public TCGA-STAD validation: PanCancer TLS × PathPrism", fontsize=15)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(outdir / f"figure1_primary_validation.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def radial_figure(
    pp_root: Path,
    positive_samples: list[str],
    outdir: Path,
    pp_analysis_name: str = "public_stad_validation_v1",
) -> pd.DataFrame:
    rings = [
        ("0–250", "ring_0_250_fraction_LYM"),
        ("250–500", "ring_250_500_fraction_LYM"),
        ("500–1000", "ring_500_1000_fraction_LYM"),
    ]
    records = []
    for sample in positive_samples:
        path = (
            pp_root / "reproduction/outputs" / pp_analysis_name / sample
            / "scale_112um/tls_tissue_graph/tls_control_features.tsv"
        )
        d = pd.read_csv(path, sep="\t")
        d = d[(d.radius_um == 1000) & (d.semantic_threshold == 0.0)]
        for oid in sorted(d.object_id.unique()):
            g = d[d.object_id == oid]
            tls = g[g.location_type == "TLS"].iloc[0]
            controls = g[g.location_type == "matched_control"]
            for ring_label, col in rings:
                records.append(
                    {
                        "sample_id": sample,
                        "object_id": int(oid),
                        "ring": ring_label,
                        "tls": float(tls[col]),
                        "control_mean": float(controls[col].mean()),
                    }
                )
    radial = pd.DataFrame(records)
    radial["paired_difference"] = radial.tls - radial.control_mean
    radial.to_csv(outdir / "radial_lym_object_values.tsv", sep="\t", index=False)

    order = [x[0] for x in rings]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for sample, color in zip(positive_samples, SLIDE_COLORS):
        g = radial[radial.sample_id == sample].groupby("ring", as_index=False).mean(numeric_only=True)
        g["ring"] = pd.Categorical(g.ring, categories=order, ordered=True)
        g = g.sort_values("ring")
        axes[0].plot(order, g.tls, "-o", color=color, label=sample.replace("TCGA-", ""))
        axes[0].plot(order, g.control_mean, "--o", color=color, alpha=0.55)
        axes[1].plot(order, g.paired_difference, "-o", color=color, label=sample.replace("TCGA-", ""))
    axes[0].set_title("Slide mean radial LYM fraction")
    axes[0].set_ylabel("LYM fraction")
    axes[0].text(0.02, 0.98, "solid: TLS  •  dashed: matched controls", transform=axes[0].transAxes, va="top", fontsize=9)
    axes[1].set_title("TLS − matched control")
    axes[1].axhline(0, color="#777777", linewidth=0.8)
    axes[1].set_ylabel("Paired difference")
    axes[1].legend(frameon=False, fontsize=8)
    for ax, label in zip(axes, ["A", "B"]):
        ax.set_xlabel("Distance from object center (µm)")
        add_panel_label(ax, label)
    fig.suptitle("Radial localization of the LYM signal (112 µm)")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(outdir / f"figure2_radial_profile.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return radial


def qc_figure(slides: pd.DataFrame, qc: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ax = axes[0]
    ax.bar(
        slides.sample_id.str.replace("TCGA-", "", regex=False),
        slides.total_tls,
        color=["#BAB0AC" if x == 0 else "#4C78A8" for x in slides.total_tls],
    )
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylabel("TLS candidates")
    ax.set_title("Two of five slides were zero-detection")
    add_panel_label(ax, "A")

    ax = axes[1]
    pos = slides[slides.total_tls > 0]
    frac = pos.stable_tls / pos.total_tls
    ax.bar(pos.sample_id.str.replace("TCGA-", "", regex=False), frac, color="#59A14F")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("10%/25% context consistency")
    ax.set_title("Maturity classification stability")
    add_panel_label(ax, "B")

    ax = axes[2]
    stable = qc.context_stable.astype(bool)
    ax.scatter(qc.s112_inner_diff[stable], qc.s256_inner_diff[stable], color="#4C78A8", label="stable", alpha=0.8)
    ax.scatter(qc.s112_inner_diff[~stable], qc.s256_inner_diff[~stable], color="#E15759", marker="x", label="uncertain")
    lim = max(abs(qc.s112_inner_diff).max(), abs(qc.s256_inner_diff).max(), 0.1) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color="#777777", linewidth=0.8)
    ax.axhline(0, color="#BBBBBB", linewidth=0.6)
    ax.axvline(0, color="#BBBBBB", linewidth=0.6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("112 µm paired Δ")
    ax.set_ylabel("256 µm paired Δ")
    ax.set_title("Object-level scale sensitivity")
    ax.legend(frameon=False)
    add_panel_label(ax, "C")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(outdir / f"figure3_qc_robustness.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def sha256_manifest(root: Path) -> None:
    lines = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name == "SHA256SUMS":
            continue
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {p.relative_to(root)}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def main() -> None:
    a = parse_args()
    if a.output_dir.exists():
        if not a.overwrite:
            raise SystemExit(f"Output exists: {a.output_dir}; use --overwrite")
        shutil.rmtree(a.output_dir)
    a.output_dir.mkdir(parents=True)
    (a.output_dir / "overviews").mkdir()
    (a.output_dir / "object_qc").mkdir()
    (a.output_dir / "contact_sheets").mkdir()
    (a.output_dir / "figures").mkdir()

    cohort = a.pp_root / "reproduction/outputs" / a.pp_cohort_name
    pp_analysis_root = a.pp_root / "reproduction/outputs" / a.pp_analysis_name
    pc_analysis_root = a.pc_root / "reproduction/outputs" / a.pc_analysis_name
    slides = pd.read_csv(cohort / "cohort_slide_summary.tsv", sep="\t")
    objects = pd.read_csv(cohort / "cohort_object_paired.tsv", sep="\t")
    records = []
    pred_cache = {}
    meta_cache = {}
    tile_cache = {}

    for slide_row in slides.itertuples():
        sample = slide_row.sample_id
        pc_dir = pc_analysis_root / sample
        pp_dir = pp_analysis_root / sample
        overview_panel(
            sample,
            pc_dir,
            pp_dir,
            a.output_dir / "overviews" / f"{sample}_overview.png",
            pd.Series(slide_row._asdict()),
        )
        if int(slide_row.total_tls) == 0:
            continue
        preds = pd.read_csv(pc_dir / "maturity_predictions.tsv", sep="\t")
        preds["object_id"] = preds.object_id.astype(int)
        pred_cache[sample] = preds.set_index("object_id")
        metadata_list = json.load(open(pc_dir / "crops/crop_metadata.json"))
        metadata = {(int(x["patch_id"]), round(float(x["padding_ratio"]), 2)): x for x in metadata_list}
        meta_cache[sample] = metadata
        tile112 = pd.read_csv(pp_dir / "scale_112um/prismnet/tile_predictions.tsv", sep="\t")
        tile_cache[sample] = tile112
        features112 = pd.read_csv(pp_dir / "scale_112um/tls_tissue_graph/tls_control_features.tsv", sep="\t")
        features112 = features112[
            (features112.radius_um == 1000)
            & (features112.semantic_threshold == 0.0)
            & (features112.location_type == "matched_control")
        ]
        for pred in preds.itertuples(index=False):
            oid = int(pred.object_id)
            poly = json.loads(pred.polygon_coordinates)
            area = polygon_area(poly)
            cx, cy = polygon_centroid(poly)
            bbox = json.loads(pred.physical_bbox_um)
            bw, bh = float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])
            p10 = pc_dir / "crops/extracted_patches_10" / f"patch_{oid}.jpeg"
            p25 = pc_dir / "crops/extracted_patches_25" / f"patch_{oid}.jpeg"
            tex10, tissue10, white10 = image_texture(p10)
            tex25, tissue25, white25 = image_texture(p25)
            obj112 = objects[(objects.sample_id == sample) & (objects.scale_um == 112) & (objects.object_id == oid)].iloc[0]
            obj256 = objects[(objects.sample_id == sample) & (objects.scale_um == 256) & (objects.object_id == oid)].iloc[0]
            center_tiles = tile112[
                ((tile112.center_x_um - cx) ** 2 + (tile112.center_y_um - cy) ** 2) <= 250**2
            ]
            cg = features112[features112.object_id == oid]
            control_match_max = float(cg.match_distance.max()) if len(cg) else float("nan")
            flags = []
            if not bool(pred.context_stable):
                flags.append("context_unstable")
            if any(bool(metadata[(oid, pad)]["clamped_to_boundary"]) for pad in [0.1, 0.25]):
                flags.append("crop_clamped")
            d112 = float(obj112.ring_0_250_fraction_LYM_paired_difference)
            d256 = float(obj256.ring_0_250_fraction_LYM_paired_difference)
            if d112 * d256 < 0:
                flags.append("scale_direction_discordant")
            min_conf = min(max_joint_conf(pd.Series(pred._asdict()), "pad10"), max_joint_conf(pd.Series(pred._asdict()), "pad25"))
            if min_conf < 0.70:
                flags.append("low_joint_confidence")
            if len(center_tiles) < 4:
                flags.append("sparse_inner_tiles")
            if min(tex10, tex25) < 4.0:
                flags.append("low_texture")
            records.append(
                {
                    "sample_id": sample,
                    "object_id": oid,
                    "consensus_class": pred.consensus_class,
                    "context_stable": bool(pred.context_stable),
                    "polygon_area_um2": area,
                    "equivalent_diameter_um": math.sqrt(4 * area / math.pi) if area else 0,
                    "bbox_width_um": bw,
                    "bbox_height_um": bh,
                    "bbox_aspect_ratio": max(bw, bh) / max(min(bw, bh), 1e-9),
                    "center_x_um": cx,
                    "center_y_um": cy,
                    "pad10_joint_confidence": max_joint_conf(pd.Series(pred._asdict()), "pad10"),
                    "pad25_joint_confidence": max_joint_conf(pd.Series(pred._asdict()), "pad25"),
                    "patch10_gradient_rms": tex10,
                    "patch25_gradient_rms": tex25,
                    "patch10_tissue_fraction": tissue10,
                    "patch25_tissue_fraction": tissue25,
                    "patch10_white_fraction": white10,
                    "patch25_white_fraction": white25,
                    "n_112_tiles_inner_250": int(len(center_tiles)),
                    "max_control_match_distance": control_match_max,
                    "s112_inner_tls": float(obj112.ring_0_250_fraction_LYM_TLS),
                    "s112_inner_control_mean": float(obj112.ring_0_250_fraction_LYM_control_mean),
                    "s112_inner_diff": d112,
                    "s112_plym_diff": float(obj112.mean_probability_LYM_paired_difference),
                    "s256_inner_diff": d256,
                    "s256_plym_diff": float(obj256.mean_probability_LYM_paired_difference),
                    "flags": ";".join(flags),
                }
            )

    qc = pd.DataFrame(records)
    area_hi = robust_high_outlier(np.log1p(qc.polygon_area_um2))
    match_hi = robust_high_outlier(qc.max_control_match_distance)
    for idx, row in qc.iterrows():
        flags = [x for x in str(row["flags"]).split(";") if x and x != "nan"]
        if math.log1p(row.polygon_area_um2) > area_hi:
            flags.append("polygon_size_high_outlier")
        if row.max_control_match_distance > match_hi:
            flags.append("control_match_high_outlier")
        qc.at[idx, "flags"] = ";".join(flags)
        high = {"crop_clamped", "scale_direction_discordant", "sparse_inner_tiles", "low_texture", "polygon_size_high_outlier"}
        if set(flags) & high:
            qc.at[idx, "auto_qc_priority"] = "high"
        elif flags:
            qc.at[idx, "auto_qc_priority"] = "medium"
        else:
            qc.at[idx, "auto_qc_priority"] = "pass"

    # Create per-object panels after flags have been finalized.
    panel_paths_by_sample: dict[str, list[Path]] = {}
    for row in qc.itertuples(index=False):
        sample = row.sample_id
        pc_dir = pc_analysis_root / sample
        wsi_path = pc_dir / f"{sample}_05mpp.tif"
        panel_path = a.output_dir / "object_qc" / f"{sample}_object_{int(row.object_id):02d}.png"
        pred_series = pred_cache[sample].loc[int(row.object_id)]
        row_series = pd.Series(row._asdict())
        row_series["_prediction"] = pred_series
        with openslide.OpenSlide(str(wsi_path)) as wsi:
            object_panel(sample, row_series, meta_cache[sample], pc_dir, tile_cache[sample], wsi, panel_path)
        panel_paths_by_sample.setdefault(sample, []).append(panel_path)

    for sample, paths in panel_paths_by_sample.items():
        make_contact_sheets(paths, a.output_dir / "contact_sheets" / sample)

    slides.to_csv(a.output_dir / "slide_qc.tsv", sep="\t", index=False)
    qc.to_csv(a.output_dir / "object_qc.tsv", sep="\t", index=False)
    main_result_figure(slides, objects, a.output_dir / "figures")
    radial = radial_figure(
        a.pp_root,
        slides.loc[slides.total_tls > 0, "sample_id"].tolist(),
        a.output_dir / "figures",
        a.pp_analysis_name,
    )
    qc_figure(slides, qc, a.output_dir / "figures")

    summary = {
        "status": "complete",
        "scope": "engineering visual QC and cross-model consistency; not pathological ground truth",
        "n_slides": int(len(slides)),
        "n_zero_detection_slides": int((slides.total_tls == 0).sum()),
        "n_objects": int(len(qc)),
        "n_context_unstable": int((~qc.context_stable.astype(bool)).sum()),
        "n_auto_qc_high": int((qc.auto_qc_priority == "high").sum()),
        "n_auto_qc_medium": int((qc.auto_qc_priority == "medium").sum()),
        "n_auto_qc_pass": int((qc.auto_qc_priority == "pass").sum()),
        "n_scale_direction_discordant": int(qc["flags"].str.contains("scale_direction_discordant").sum()),
        "n_crop_clamped": int(qc["flags"].str.contains("crop_clamped").sum()),
        "n_low_joint_confidence": int(qc["flags"].str.contains("low_joint_confidence").sum()),
        "n_radial_records": int(len(radial)),
    }
    (a.output_dir / "qc_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    sha256_manifest(a.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
