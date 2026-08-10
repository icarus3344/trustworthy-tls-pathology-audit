#!/usr/bin/env python3
"""Object-level TLS/tissue spatial features with explicit graph definitions."""

from __future__ import annotations

import argparse
import ast
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.measure import label as sk_label
from skimage.measure import regionprops
from skimage.morphology import disk, erosion
from skimage.util import view_as_blocks

CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
NON_BACK = [c for c in CLASSES if c != "BACK"]
MATCH_CLASSES = ["ADI", "LYM", "MUS", "NORM", "STR", "TUM"]


def polygon_center_um(
    points: np.ndarray,
    mpp: float,
    method: str = "area_centroid",
) -> tuple[float, float]:
    """Return a polygon centre in micrometres.

    ``area_centroid`` uses the signed shoelace centroid and is invariant to
    uneven vertex sampling along straight edges. Degenerate polygons fall back
    to the vertex mean. ``vertex_mean`` is retained only to reproduce the
    superseded validation-v1 calculation.
    """
    xy = np.asarray(points, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) == 0:
        raise ValueError(f"Expected a non-empty Nx2 polygon, got {xy.shape}")
    if method == "vertex_mean" or len(xy) < 3:
        center = xy.mean(axis=0)
        return float(center[0] * mpp), float(center[1] * mpp)
    if method != "area_centroid":
        raise ValueError(f"Unsupported TLS centre method: {method}")

    x, y = xy[:, 0], xy[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    cross_sum = float(cross.sum())
    if abs(cross_sum) < 1e-12:
        center = xy.mean(axis=0)
    else:
        center = np.asarray(
            [
                ((x + np.roll(x, -1)) * cross).sum() / (3.0 * cross_sum),
                ((y + np.roll(y, -1)) * cross).sum() / (3.0 * cross_sum),
            ]
        )
    return float(center[0] * mpp), float(center[1] * mpp)


def entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    if counts.sum() == 0:
        return float("nan")
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def build_label_grid(
    tiles: pd.DataFrame, threshold: float, tile_um: float
) -> tuple[np.ndarray, np.ndarray]:
    gx = np.rint((tiles.center_x_um.to_numpy() - tile_um / 2) / tile_um).astype(int)
    gy = np.rint((tiles.center_y_um.to_numpy() - tile_um / 2) / tile_um).astype(int)
    if gx.min() < 0 or gy.min() < 0:
        raise ValueError("Negative tile grid coordinate")
    grid = np.full((gy.max() + 1, gx.max() + 1), -1, dtype=np.int16)
    valid = tiles.confidence.to_numpy() >= threshold
    labs = np.asarray([CLASSES.index(x) for x in tiles.predicted_class], dtype=np.int16)
    grid[gy[valid], gx[valid]] = labs[valid]
    return grid, np.c_[gx, gy]


def build_probability_grid(tiles: pd.DataFrame, tile_um: float) -> np.ndarray:
    gx = np.rint((tiles.center_x_um.to_numpy() - tile_um / 2) / tile_um).astype(int)
    gy = np.rint((tiles.center_y_um.to_numpy() - tile_um / 2) / tile_um).astype(int)
    grid = np.full((gy.max() + 1, gx.max() + 1, len(CLASSES)), np.nan, dtype=np.float32)
    grid[gy, gx] = tiles[[f"p_{c}" for c in CLASSES]].to_numpy(np.float32)
    return grid


def components_for_class(subgrid: np.ndarray, class_name: str) -> tuple[np.ndarray, int, list[int]]:
    labels, n = ndimage.label(
        subgrid == CLASSES.index(class_name), structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    )
    sizes = np.bincount(labels.ravel())[1:].astype(int).tolist() if n else []
    return labels, int(n), sizes


def boundary_pair_features(subgrid: np.ndarray, a: str, b: str, tile_um: float) -> dict:
    la, na, _ = components_for_class(subgrid, a)
    lb, nb, _ = components_for_class(subgrid, b)
    pairs: set[tuple[int, int]] = set()
    edge_count = 0
    # horizontal shared sides
    left, right = subgrid[:, :-1], subgrid[:, 1:]
    for mask, a_labels, b_labels in [
        ((left == CLASSES.index(a)) & (right == CLASSES.index(b)), la[:, :-1], lb[:, 1:]),
        ((left == CLASSES.index(b)) & (right == CLASSES.index(a)), la[:, 1:], lb[:, :-1]),
    ]:
        ys, xs = np.where(mask)
        edge_count += len(ys)
        pairs.update((int(a_labels[y, x]), int(b_labels[y, x])) for y, x in zip(ys, xs))
    # vertical shared sides
    top, bottom = subgrid[:-1, :], subgrid[1:, :]
    for mask, a_labels, b_labels in [
        ((top == CLASSES.index(a)) & (bottom == CLASSES.index(b)), la[:-1, :], lb[1:, :]),
        ((top == CLASSES.index(b)) & (bottom == CLASSES.index(a)), la[1:, :], lb[:-1, :]),
    ]:
        ys, xs = np.where(mask)
        edge_count += len(ys)
        pairs.update((int(a_labels[y, x]), int(b_labels[y, x])) for y, x in zip(ys, xs))
    pairs.discard((0, 0))
    pairs = {(x, y) for x, y in pairs if x > 0 and y > 0}
    return {
        f"{a.lower()}_{b.lower()}_contact_pair_count": len(pairs),
        f"{a.lower()}_{b.lower()}_interface_um": float(edge_count * tile_um),
        f"{a.lower()}_{b.lower()}_component_pair_ids": sorted([list(x) for x in pairs]),
    }


def proximity_pair_features(
    subgrid: np.ndarray, a: str, b: str, max_distance_cells: int = 2, tile_um: float = 256
) -> dict:
    """Count component pairs within a named coarse-grid proximity threshold."""
    la, na, _ = components_for_class(subgrid, a)
    lb, nb, _ = components_for_class(subgrid, b)
    pairs: set[tuple[int, int]] = set()
    footprint = disk(max_distance_cells).astype(bool)
    for aid in range(1, na + 1):
        nearby = ndimage.binary_dilation(la == aid, structure=footprint)
        for bid in np.unique(lb[nearby]):
            if bid > 0:
                pairs.add((aid, int(bid)))
    distance_um = int(round(max_distance_cells * tile_um))
    key = f"{a.lower()}_{b.lower()}_proximity_pair_count_{distance_um}um"
    return {key: len(pairs)}


def lym_bridge_count(subgrid: np.ndarray) -> int:
    ll, nl, _ = components_for_class(subgrid, "LYM")
    touched_tum, touched_str = set(), set()
    for dy, dx in [(0, 1), (1, 0)]:
        s1 = (slice(None, -dy or None), slice(None, -dx or None))
        s2 = (slice(dy, None), slice(dx, None))
        g1, g2 = subgrid[s1], subgrid[s2]
        l1, l2 = ll[s1], ll[s2]
        for target, out in [("TUM", touched_tum), ("STR", touched_str)]:
            t = CLASSES.index(target)
            ys, xs = np.where((g1 == CLASSES.index("LYM")) & (g2 == t))
            out.update(int(l1[y, x]) for y, x in zip(ys, xs) if l1[y, x] > 0)
            ys, xs = np.where((g2 == CLASSES.index("LYM")) & (g1 == t))
            out.update(int(l2[y, x]) for y, x in zip(ys, xs) if l2[y, x] > 0)
    return len(touched_tum & touched_str)


def official_region_centroids(
    subgrid: np.ndarray,
    class_name: str,
    erosion_radius: float = 1.5,
    area_thresh: int = 2000,
    block_size: int = 30,
) -> np.ndarray:
    """Match upstream PathPrism region-node creation, including its <5-pixel gate."""
    mask = (subgrid == CLASSES.index(class_name)).astype(np.uint8)
    if mask.sum() < 5:
        return np.empty((0, 2), dtype=float)
    if erosion_radius > 0:
        mask = erosion(mask, disk(erosion_radius))
    labelled = sk_label(mask)
    nodes: list[tuple[float, float]] = []
    for region in regionprops(labelled):
        if region.area > area_thresh:
            minr, minc, maxr, maxc = region.bbox
            region_label = labelled[minr:maxr, minc:maxc] == region.label
            h, w = region_label.shape
            pad_h = (block_size - h % block_size) % block_size
            pad_w = (block_size - w % block_size) % block_size
            padded = np.pad(region_label, ((0, pad_h), (0, pad_w)), mode="constant")
            blocks = view_as_blocks(padded, block_shape=(block_size, block_size))
            for i in range(blocks.shape[0]):
                for j in range(blocks.shape[1]):
                    if blocks[i, j].mean() > 0.5:
                        # Upstream stores node pos as (x, y).
                        nodes.append(
                            (
                                float(minc + j * block_size + block_size // 2),
                                float(minr + i * block_size + block_size // 2),
                            )
                        )
        else:
            cy, cx = region.centroid
            nodes.append((float(cx), float(cy)))
    return np.asarray(nodes, dtype=float).reshape(-1, 2)


def centroid_pair_features(
    subgrid: np.ndarray, a: str, b: str, threshold_cells: float = 20.0
) -> dict:
    ca, cb = official_region_centroids(subgrid, a), official_region_centroids(subgrid, b)
    if len(ca) and len(cb):
        dist = np.sqrt(((ca[:, None] - cb[None, :]) ** 2).sum(2))
        edges = dist < threshold_cells
        n_edges = int(edges.sum())
        density = float(n_edges / edges.size)
        mean_dist = float(dist[edges].mean()) if n_edges else float("nan")
    else:
        n_edges, density, mean_dist = 0, 0.0, float("nan")
    key = f"official_{a.lower()}_{b.lower()}"
    return {
        f"{key}_n_nodes1": int(len(ca)),
        f"{key}_n_nodes2": int(len(cb)),
        f"{key}_n_inter_edges": n_edges,
        f"{key}_inter_edge_density": density,
        f"{key}_mean_inter_distance_cells": mean_dist,
        f"{key}_distance_threshold_cells": threshold_cells,
        f"{key}_erosion_radius_cells": 1.5,
        f"{key}_large_region_split_threshold_cells": 2000,
        f"{key}_block_size_cells": 30,
    }


def extract_window(
    grid: np.ndarray, cx_um: float, cy_um: float, radius_um: float, tile_um: float
) -> tuple[np.ndarray, tuple[float, float]]:
    cx, cy = cx_um / tile_um - 0.5, cy_um / tile_um - 0.5
    r = radius_um / tile_um
    x0, x1 = max(0, int(math.floor(cx - r))), min(grid.shape[1], int(math.ceil(cx + r)) + 1)
    y0, y1 = max(0, int(math.floor(cy - r))), min(grid.shape[0], int(math.ceil(cy + r)) + 1)
    sub = grid[y0:y1, x0:x1].copy()
    yy, xx = np.mgrid[y0:y1, x0:x1]
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
    sub[~circle] = -1
    return sub, (cx - x0, cy - y0)


def window_features(
    grid: np.ndarray,
    cx_um: float,
    cy_um: float,
    radius_um: float,
    tile_um: float,
    probability_grid: np.ndarray | None = None,
) -> dict:
    sub, (local_cx, local_cy) = extract_window(grid, cx_um, cy_um, radius_um, tile_um)
    vals = sub[sub >= 0]
    nonback = vals[vals != CLASSES.index("BACK")]
    out = {"n_valid_tiles": int(len(vals)), "n_nonback_tiles": int(len(nonback))}
    counts = np.asarray([np.sum(nonback == CLASSES.index(c)) for c in NON_BACK])
    for c, n in zip(NON_BACK, counts):
        out[f"fraction_{c}"] = float(n / len(nonback)) if len(nonback) else float("nan")
    out["semantic_entropy"] = entropy_from_counts(counts)
    if probability_grid is not None:
        cx, cy = cx_um / tile_um - 0.5, cy_um / tile_um - 0.5
        r = radius_um / tile_um
        x0, x1 = max(0, int(math.floor(cx - r))), min(grid.shape[1], int(math.ceil(cx + r)) + 1)
        y0, y1 = max(0, int(math.floor(cy - r))), min(grid.shape[0], int(math.ceil(cy + r)) + 1)
        psub = probability_grid[y0:y1, x0:x1]
        pkeep = (sub >= 0) & (sub != CLASSES.index("BACK")) & np.isfinite(psub[..., 0])
        for i, c in enumerate(CLASSES):
            out[f"mean_probability_{c}"] = (
                float(psub[..., i][pkeep].mean()) if pkeep.any() else float("nan")
            )
    yy, xx = np.mgrid[: sub.shape[0], : sub.shape[1]]
    dist_um = np.hypot(xx - local_cx, yy - local_cy) * tile_um
    if radius_um >= 1000:
        for inner, outer in [(0, 250), (250, 500), (500, 1000)]:
            ring_vals = sub[(dist_um >= inner) & (dist_um < outer)]
            ring_nonback = ring_vals[(ring_vals >= 0) & (ring_vals != CLASSES.index("BACK"))]
            for c in NON_BACK:
                out[f"ring_{inner}_{outer}_fraction_{c}"] = (
                    float(np.mean(ring_nonback == CLASSES.index(c)))
                    if len(ring_nonback)
                    else float("nan")
                )
    for c in ["LYM", "STR", "TUM"]:
        _, n, sizes = components_for_class(sub, c)
        out[f"component_count_{c}"] = n
        out[f"largest_component_tiles_{c}"] = max(sizes, default=0)
        ys, xs = np.where(sub == CLASSES.index(c))
        if len(xs):
            out[f"nearest_{c}_um"] = float(np.min(np.hypot(xs - local_cx, ys - local_cy)) * tile_um)
        else:
            out[f"nearest_{c}_um"] = float(radius_um)
    out.update(boundary_pair_features(sub, "LYM", "TUM", tile_um))
    out.update(boundary_pair_features(sub, "LYM", "STR", tile_um))
    # Two 256-um grid cells under the frozen primary configuration.
    out.update(proximity_pair_features(sub, "LYM", "TUM", 2, tile_um))
    out.update(proximity_pair_features(sub, "LYM", "STR", 2, tile_um))
    out["lym_bridge_component_count"] = lym_bridge_count(sub)
    out.update(centroid_pair_features(sub, "LYM", "TUM"))
    out.update(centroid_pair_features(sub, "LYM", "STR"))
    return out


def broad_vector(grid: np.ndarray, x: float, y: float, radius: float, tile_um: float) -> np.ndarray:
    f = window_features(grid, x, y, radius, tile_um)
    return np.asarray([f[f"fraction_{c}"] for c in MATCH_CLASSES], dtype=float)


def render_locations(
    wsi_path: Path,
    wsi_mpp: float,
    tls: pd.DataFrame,
    controls: list[dict],
    output: Path,
    radius_um: float = 1000,
) -> None:
    import openslide
    from PIL import ImageDraw

    slide = openslide.OpenSlide(str(wsi_path))
    thumb = slide.get_thumbnail((1800, 1800)).convert("RGB")
    sx, sy = thumb.width / slide.dimensions[0], thumb.height / slide.dimensions[1]
    draw = ImageDraw.Draw(thumb)
    for _, row in tls.iterrows():
        x = row.center_x_um / wsi_mpp * sx
        y = row.center_y_um / wsi_mpp * sy
        rx, ry = radius_um / wsi_mpp * sx, radius_um / wsi_mpp * sy
        draw.ellipse((x - rx, y - ry, x + rx, y + ry), outline=(255, 0, 0), width=3)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 0, 0))
        draw.text(
            (x + 6, y + 2),
            f"TLS {row.object_id}",
            fill=(0, 0, 0),
            stroke_width=2,
            stroke_fill="white",
        )
    for c in controls:
        x = c["center_x_um"] / wsi_mpp * sx
        y = c["center_y_um"] / wsi_mpp * sy
        rx, ry = radius_um / wsi_mpp * sx, radius_um / wsi_mpp * sy
        draw.ellipse((x - rx, y - ry, x + rx, y + ry), outline=(0, 220, 255), width=3)
        draw.rectangle((x - 5, y - 5, x + 5, y + 5), fill=(0, 220, 255))
        draw.text(
            (x + 6, y + 2),
            f"C {c['object_id']}.{c.get('control_id', 1)}",
            fill=(0, 0, 0),
            stroke_width=2,
            stroke_fill="white",
        )
    thumb.save(output)
    slide.close()


def select_controls(
    grid: np.ndarray,
    tls: pd.DataFrame,
    tile_um: float,
    radius_um: float,
    exclusion_um: float,
    controls_per_tls: int = 1,
) -> list[dict]:
    ys, xs = np.where((grid >= 0) & (grid != CLASSES.index("BACK")))
    step_cells = max(1, int(round(radius_um / tile_um)))
    keep = (xs % step_cells == 0) & (ys % step_cells == 0)
    candidates = np.c_[(xs[keep] + 0.5) * tile_um, (ys[keep] + 0.5) * tile_um]
    tls_xy = tls[["center_x_um", "center_y_um"]].to_numpy(float)
    if len(candidates):
        far = (
            np.min(np.sqrt(((candidates[:, None] - tls_xy[None, :]) ** 2).sum(2)), axis=1)
            >= exclusion_um
        )
        candidates = candidates[far]
    if not len(candidates):
        raise ValueError("No eligible matched-control candidates")
    cv = np.vstack([broad_vector(grid, x, y, radius_um * 1.5, tile_um) for x, y in candidates])
    cv = np.nan_to_num(cv, nan=0.0)
    scale = np.nanstd(cv, axis=0)
    scale[scale == 0] = 1
    used: set[int] = set()
    controls = []
    for _, row in tls.iterrows():
        tv = np.nan_to_num(
            broad_vector(grid, row.center_x_um, row.center_y_um, radius_um * 1.5, tile_um), nan=0.0
        )
        d = np.sqrt((((cv - tv) / scale) ** 2).sum(1))
        order = np.argsort(d)
        for control_id in range(1, controls_per_tls + 1):
            idx = next(
                (int(i) for i in order if int(i) not in used),
                int(order[(control_id - 1) % len(order)]),
            )
            reused = idx in used
            used.add(idx)
            controls.append(
                {
                    "object_id": str(row.object_id),
                    "control_id": control_id,
                    "center_x_um": float(candidates[idx, 0]),
                    "center_y_um": float(candidates[idx, 1]),
                    "match_distance": float(d[idx]),
                    "control_reused": reused,
                }
            )
    return controls


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tile-predictions", type=Path, required=True)
    p.add_argument("--tls-predictions", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--tile-um", type=float, default=256.0)
    p.add_argument("--tls-mpp", type=float, default=0.5)
    p.add_argument("--radii-um", default="500,1000,1500")
    p.add_argument("--thresholds", default="0,0.5,0.7")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--wsi", type=Path)
    p.add_argument("--wsi-mpp", type=float, default=0.5)
    p.add_argument("--controls-per-tls", type=int, default=1)
    p.add_argument(
        "--tls-center-method",
        choices=["area_centroid", "vertex_mean"],
        default="area_centroid",
        help="Polygon centre used for TLS windows; vertex_mean is legacy-only.",
    )
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = a.output_dir / "tls_control_features.tsv"
    if out_tsv.exists() and not a.overwrite:
        raise SystemExit("Output exists; use --overwrite")
    tiles = pd.read_csv(a.tile_predictions, sep="\t")
    tls = pd.read_csv(a.tls_predictions, sep="\t")
    if len(tls) == 0:
        pd.DataFrame(
            columns=[
                "sample_id",
                "object_id",
                "location_type",
                "control_id",
                "center_x_um",
                "center_y_um",
                "radius_um",
                "semantic_threshold",
            ]
        ).to_csv(out_tsv, sep="\t", index=False)
        pd.DataFrame(columns=["object_id"]).to_csv(
            a.output_dir / "primary_paired_differences.tsv", sep="\t", index=False
        )
        (a.output_dir / "pilot_summary.json").write_text(
            json.dumps(
                {
                    "status": "success",
                    "analysis_label": "zero-detection slide retained in validation cohort",
                    "n_tls_objects": 0,
                    "zero_detection": True,
                    "controls_per_tls": a.controls_per_tls,
                    "tls_center_method": a.tls_center_method,
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps({"status": "success", "objects": 0, "zero_detection": True}))
        return
    for col in ["object_id", "polygon_coordinates", "context_stable", "consensus_class"]:
        if col not in tls:
            raise ValueError(f"Missing TLS column: {col}")
    centers = []
    for s in tls.polygon_coordinates:
        xy = np.asarray(ast.literal_eval(s), dtype=float)
        centers.append(polygon_center_um(xy, a.tls_mpp, a.tls_center_method))
    tls["center_x_um"] = [x[0] for x in centers]
    tls["center_y_um"] = [x[1] for x in centers]
    radii = [float(x) for x in a.radii_um.split(",")]
    thresholds = [float(x) for x in a.thresholds.split(",")]
    records = []
    control_manifest = []
    primary_controls = None
    prob_grid = build_probability_grid(tiles, a.tile_um)
    for threshold in thresholds:
        grid, _ = build_label_grid(tiles, threshold, a.tile_um)
        controls = select_controls(
            grid,
            tls,
            a.tile_um,
            radius_um=1000,
            exclusion_um=1500,
            controls_per_tls=a.controls_per_tls,
        )
        if threshold == 0:
            primary_controls = controls
        control_manifest.extend([{**x, "semantic_threshold": threshold} for x in controls])
        controls_by_id: dict[str, list[dict]] = defaultdict(list)
        for control in controls:
            controls_by_id[str(control["object_id"])].append(control)
        for _, row in tls.iterrows():
            locations = [
                ("TLS", 0, float(row.center_x_um), float(row.center_y_um), float("nan")),
            ]
            locations.extend(
                [
                    (
                        "matched_control",
                        int(control["control_id"]),
                        control["center_x_um"],
                        control["center_y_um"],
                        control["match_distance"],
                    )
                    for control in controls_by_id[str(row.object_id)]
                ]
            )
            for location_type, control_id, x, y, match_distance in locations:
                for radius in radii:
                    feat = window_features(grid, x, y, radius, a.tile_um, prob_grid)
                    records.append(
                        {
                            "sample_id": row.sample_id,
                            "object_id": row.object_id,
                            "location_type": location_type,
                            "control_id": control_id,
                            "center_x_um": x,
                            "center_y_um": y,
                            "radius_um": radius,
                            "semantic_threshold": threshold,
                            "match_distance": match_distance,
                            "tls_center_method": a.tls_center_method,
                            "context_stable": bool(row.context_stable),
                            "consensus_class": row.consensus_class,
                            **{k: v for k, v in feat.items() if not k.endswith("_ids")},
                        }
                    )
    result = pd.DataFrame(records)
    result.to_csv(out_tsv, sep="\t", index=False)
    primary = result[(result.radius_um == 1000) & (result.semantic_threshold == 0)]
    paired = []
    proximity_um = int(round(2 * a.tile_um))
    tum_proximity_key = f"lym_tum_proximity_pair_count_{proximity_um}um"
    str_proximity_key = f"lym_str_proximity_pair_count_{proximity_um}um"
    numeric_features = [
        "fraction_LYM",
        "mean_probability_LYM",
        "fraction_STR",
        "fraction_TUM",
        "component_count_LYM",
        "lym_tum_contact_pair_count",
        "lym_str_contact_pair_count",
        "lym_tum_interface_um",
        "lym_str_interface_um",
        "lym_bridge_component_count",
        tum_proximity_key,
        str_proximity_key,
        "ring_0_250_fraction_LYM",
        "ring_0_250_fraction_TUM",
        "ring_0_250_fraction_STR",
    ]
    for obj in sorted(primary.object_id.astype(str).unique()):
        q = primary[primary.object_id.astype(str) == obj]
        tls_rows = q[q.location_type == "TLS"]
        control_rows = q[q.location_type == "matched_control"]
        if len(tls_rows) == 1 and len(control_rows) >= 1:
            tls_row = tls_rows.iloc[0]
            rec = {"object_id": obj, "n_controls": int(len(control_rows))}
            for f in numeric_features:
                control_mean = control_rows[f].mean()
                rec[f"{f}_TLS"] = tls_row[f]
                rec[f"{f}_control_mean"] = control_mean
                rec[f"{f}_paired_difference"] = tls_row[f] - control_mean
            paired.append(rec)
    paired_df = pd.DataFrame(paired)
    paired_df.to_csv(a.output_dir / "primary_paired_differences.tsv", sep="\t", index=False)
    summary = {
        "status": "success",
        "analysis_label": "single-slide cross-model consistency pilot; not accuracy or biological validation",
        "n_tls_objects": int(len(tls)),
        "n_feature_rows": int(len(result)),
        "radii_um": radii,
        "semantic_thresholds": thresholds,
        "primary_radius_um": 1000,
        "primary_threshold": 0,
        "controls_per_tls": a.controls_per_tls,
        "tls_center_method": a.tls_center_method,
        "matched_controls": control_manifest,
        "primary_median_paired_differences": {
            f: float(paired_df[f"{f}_paired_difference"].median()) for f in numeric_features
        }
        if len(paired_df)
        else {},
    }
    tls_inner = primary[primary.location_type == "TLS"]["ring_0_250_fraction_LYM"]
    control_inner = (
        primary[primary.location_type == "matched_control"]
        .groupby("object_id")["ring_0_250_fraction_LYM"]
        .mean()
    )
    summary["primary_inner_250um_lym"] = {
        "tls_positive_count": int((tls_inner.fillna(0) > 0).sum()),
        "tls_total": int(len(tls_inner)),
        "control_positive_count": int((control_inner.fillna(0) > 0).sum()),
        "control_total": int(len(control_inner)),
        "tls_median_fraction": float(tls_inner.median()),
        "control_median_fraction": float(control_inner.median()),
    }
    tls_primary = primary[primary.location_type == "TLS"]
    summary["tls_primary_by_context_stability"] = {}
    for stable, group in tls_primary.groupby("context_stable"):
        summary["tls_primary_by_context_stability"][str(bool(stable))] = {
            "n": int(len(group)),
            "median_fraction_LYM": float(group.fraction_LYM.median()),
            "median_fraction_TUM": float(group.fraction_TUM.median()),
            "median_fraction_STR": float(group.fraction_STR.median()),
            f"median_lym_tum_proximity_pairs_{proximity_um}um": float(
                group[tum_proximity_key].median()
            ),
            f"median_lym_str_proximity_pairs_{proximity_um}um": float(
                group[str_proximity_key].median()
            ),
        }
    (a.output_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if a.wsi and primary_controls is not None:
        render_locations(
            a.wsi, a.wsi_mpp, tls, primary_controls, a.output_dir / "tls_control_overlay.png"
        )
    print(json.dumps({"status": "success", "rows": len(result), "objects": len(tls)}))


if __name__ == "__main__":
    main()
