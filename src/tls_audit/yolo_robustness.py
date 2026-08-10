#!/usr/bin/env python3
"""Interpretability and robustness audit for the reconstructed TLS YOLO cascade.

The script is deliberately read-only with respect to existing model outputs.  It
re-crops the same HookNet polygons into an isolated output directory and runs:

* Grad-CAM for Step 1 and Step 2;
* occlusion sensitivity for the derived E/P/S score;
* TLS-only and context-only blurred counterfactuals;
* 0/5/10/15/25/40% padding curves;
* stain, colour and encoding sensitivity.

The E/P/S probabilities are an explicitly labelled engineering derivation:
  p(E)=p1(I)/(p1(I)+p1(M))
  p(P)=p1(M)/(p1(I)+p1(M))*p2(P)
  p(S)=p1(M)/(p1(I)+p1(M))*p2(S)
The unpublished Step-1 Primary/Secondary probability mass is preserved in the
output, but is not silently folded into these derived scores.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
import PIL
from PIL import Image, ImageDraw
import torch
from ultralytics import YOLO


PADDINGS = [0.0, 0.05, 0.10, 0.15, 0.25, 0.40]
FINAL_CLASSES = ["E-TLS", "P-TLS", "S-TLS"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def entropy_and_margin(probabilities: dict[str, float]) -> tuple[float, float]:
    p = np.array([probabilities[k] for k in FINAL_CLASSES], dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    p /= p.sum()
    entropy = float(-(p * np.log(p)).sum() / np.log(len(p)))
    order = np.sort(p)
    margin = float(order[-1] - order[-2])
    return entropy, margin


def tissue_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return (gray < 238) & ((hsv[:, :, 1] > 8) | (gray < 210))


def reinhard_reference(rgb: np.ndarray) -> dict[str, list[float]]:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    mask = tissue_mask(rgb)
    values = lab[mask] if mask.sum() >= 100 else lab.reshape(-1, 3)
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": np.maximum(values.std(axis=0), 1.0).tolist(),
    }


def reinhard_normalize(rgb: np.ndarray, reference: dict[str, list[float]]) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    mask = tissue_mask(rgb)
    values = lab[mask] if mask.sum() >= 100 else lab.reshape(-1, 3)
    src_mean = values.mean(axis=0)
    src_std = np.maximum(values.std(axis=0), 1.0)
    ref_mean = np.array(reference["mean"], dtype=np.float32)
    ref_std = np.array(reference["std"], dtype=np.float32)
    transformed = (lab - src_mean) / src_std * ref_std + ref_mean
    transformed = np.clip(transformed, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(transformed, cv2.COLOR_LAB2RGB)
    out[~mask] = rgb[~mask]
    return out


def _stain_matrix(rgb: np.ndarray, alpha: float = 1.0, beta: float = 0.15) -> np.ndarray:
    flat = rgb.reshape(-1, 3).astype(np.float64)
    od = -np.log((flat + 1.0) / 256.0)
    keep = np.all(od > beta, axis=1)
    odhat = od[keep]
    if len(odhat) < 100:
        raise ValueError("Insufficient stained pixels for Macenko normalization")
    _, eigvecs = np.linalg.eigh(np.cov(odhat, rowvar=False))
    eigvecs = eigvecs[:, [2, 1]]
    projection = odhat @ eigvecs
    angles = np.arctan2(projection[:, 1], projection[:, 0])
    lo, hi = np.percentile(angles, [alpha, 100.0 - alpha])
    v1 = eigvecs @ np.array([np.cos(lo), np.sin(lo)])
    v2 = eigvecs @ np.array([np.cos(hi), np.sin(hi)])
    he = np.stack([v1, v2], axis=1)
    if he[0, 0] < he[0, 1]:
        he = he[:, [1, 0]]
    he /= np.linalg.norm(he, axis=0, keepdims=True)
    return he


def macenko_reference(rgb: np.ndarray) -> dict[str, list[list[float]] | list[float]]:
    he = _stain_matrix(rgb)
    od = -np.log((rgb.reshape(-1, 3).astype(np.float64) + 1.0) / 256.0)
    concentrations = np.linalg.lstsq(he, od.T, rcond=None)[0]
    max_c = np.maximum(np.percentile(concentrations, 99, axis=1), 1e-6)
    return {"stain_matrix": he.tolist(), "max_concentration": max_c.tolist()}


def macenko_normalize(
    rgb: np.ndarray, reference: dict[str, list[list[float]] | list[float]]
) -> np.ndarray:
    source_he = _stain_matrix(rgb)
    target_he = np.asarray(reference["stain_matrix"], dtype=np.float64)
    target_max = np.asarray(reference["max_concentration"], dtype=np.float64)
    od = -np.log((rgb.reshape(-1, 3).astype(np.float64) + 1.0) / 256.0)
    concentrations = np.linalg.lstsq(source_he, od.T, rcond=None)[0]
    source_max = np.maximum(np.percentile(concentrations, 99, axis=1), 1e-6)
    concentrations *= (target_max / source_max)[:, None]
    reconstructed = 256.0 * np.exp(-(target_he @ concentrations))
    out = np.clip(reconstructed.T.reshape(rgb.shape), 0, 255).astype(np.uint8)
    out[~tissue_mask(rgb)] = rgb[~tissue_mask(rgb)]
    return out


def encode_roundtrip(rgb: np.ndarray, fmt: str, quality: int | None = None) -> np.ndarray:
    buffer = io.BytesIO()
    kwargs = {"format": fmt}
    if quality is not None:
        kwargs["quality"] = quality
    Image.fromarray(rgb).save(buffer, **kwargs)
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGB"))


def hue_shift(rgb: np.ndarray, fraction: float) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    hsv[:, :, 0] = ((hue + round(fraction * 180)) % 180).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def saturation_scale(rgb: np.ndarray, factor: float) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def brightness_scale(rgb: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(rgb.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def crop_geometry(
    bbox: list[float], padding: float, slide_size: tuple[int, int]
) -> tuple[int, int, int, int, bool]:
    min_x, min_y, max_x, max_y = bbox
    width = max_x - min_x
    height = max_y - min_y
    pad_x = width * padding
    pad_y = height * padding
    crop_x = max(0, int(min_x - pad_x))
    crop_y = max(0, int(min_y - pad_y))
    crop_w = int(width + 2 * pad_x)
    crop_h = int(height + 2 * pad_y)
    if crop_w > crop_h:
        diff = crop_w - crop_h
        crop_y = max(0, int(crop_y - diff / 2))
        crop_h = crop_w
    elif crop_h > crop_w:
        diff = crop_h - crop_w
        crop_x = max(0, int(crop_x - diff / 2))
        crop_w = crop_h
    slide_w, slide_h = slide_size
    if crop_x + crop_w > slide_w:
        crop_w = slide_w - crop_x
    if crop_y + crop_h > slide_h:
        crop_h = slide_h - crop_y
    clamped = crop_x == 0 or crop_y == 0 or crop_x + crop_w == slide_w or crop_y + crop_h == slide_h
    if crop_w <= 0 or crop_h <= 0:
        raise ValueError(f"Invalid crop geometry: {(crop_x, crop_y, crop_w, crop_h)}")
    return crop_x, crop_y, crop_w, crop_h, clamped


def polygon_mask(
    polygon: list[list[float]], origin: tuple[int, int], size: tuple[int, int]
) -> np.ndarray:
    crop_x, crop_y = origin
    width, height = size
    points = np.array(
        [[round(x - crop_x), round(y - crop_y)] for x, y in polygon], dtype=np.int32
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    return mask


def crop_object(
    slide: openslide.OpenSlide, obj: dict, padding: float
) -> tuple[np.ndarray, np.ndarray, dict]:
    x, y, width, height, clamped = crop_geometry(
        obj["level0_bbox"], padding, slide.dimensions
    )
    rgb = np.asarray(slide.read_region((x, y), 0, (width, height)).convert("RGB"))
    mask = polygon_mask(obj["polygon_coordinates"], (x, y), (width, height))
    return rgb, mask, {
        "crop_x": x,
        "crop_y": y,
        "crop_width": width,
        "crop_height": height,
        "clamped": clamped,
    }


class CascadeModels:
    def __init__(self, step1: Path, step2: Path, device: str):
        self.step1_path = step1
        self.step2_path = step2
        self.yolo1 = YOLO(str(step1))
        self.yolo2 = YOLO(str(step2))
        self.net1 = self.yolo1.model.to(device).eval()
        self.net2 = self.yolo2.model.to(device).eval()
        self.device = torch.device(device)
        self.names1 = dict(self.yolo1.names)
        self.names2 = dict(self.yolo2.names)
        if set(self.names1.values()) != {"Immature", "Mature", "Primary", "Secondary"}:
            raise ValueError(f"Unexpected Step-1 names: {self.names1}")
        if set(self.names2.values()) != {"Primary", "Secondary"}:
            raise ValueError(f"Unexpected Step-2 names: {self.names2}")

    def tensors(self, images: list[np.ndarray], model_number: int) -> torch.Tensor:
        net = self.net1 if model_number == 1 else self.net2
        transformed = [net.transforms(Image.fromarray(x)) for x in images]
        return torch.stack(transformed).to(self.device)

    @torch.inference_mode()
    def raw_probabilities(
        self, images: list[np.ndarray], batch_size: int = 32
    ) -> tuple[np.ndarray, np.ndarray]:
        out1, out2 = [], []
        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            prediction1 = self.net1(self.tensors(chunk, 1))
            prediction2 = self.net2(self.tensors(chunk, 2))
            if isinstance(prediction1, (tuple, list)):
                prediction1 = prediction1[0]
            if isinstance(prediction2, (tuple, list)):
                prediction2 = prediction2[0]
            out1.append(prediction1.float().cpu().numpy())
            out2.append(prediction2.float().cpu().numpy())
        return np.concatenate(out1), np.concatenate(out2)

    def preprocessed_rgb(self, image: np.ndarray) -> np.ndarray:
        """Return the exact 640x640 float RGB array seen by the checkpoint."""
        tensor = self.net1.transforms(Image.fromarray(image))
        return tensor.permute(1, 2, 0).cpu().numpy()

    @torch.inference_mode()
    def raw_probabilities_preprocessed(
        self, images: list[np.ndarray], batch_size: int = 32
    ) -> tuple[np.ndarray, np.ndarray]:
        out1, out2 = [], []
        for start in range(0, len(images), batch_size):
            chunk = np.stack(images[start : start + batch_size], axis=0)
            tensor = (
                torch.from_numpy(chunk)
                .permute(0, 3, 1, 2)
                .contiguous()
                .float()
                .to(self.device)
            )
            prediction1 = self.net1(tensor)
            prediction2 = self.net2(tensor)
            if isinstance(prediction1, (tuple, list)):
                prediction1 = prediction1[0]
            if isinstance(prediction2, (tuple, list)):
                prediction2 = prediction2[0]
            out1.append(prediction1.float().cpu().numpy())
            out2.append(prediction2.float().cpu().numpy())
        return np.concatenate(out1), np.concatenate(out2)

    def derived(
        self, p1: np.ndarray, p2: np.ndarray
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], str, str]:
        raw1 = {self.names1[i]: float(p1[i]) for i in range(len(p1))}
        raw2 = {self.names2[i]: float(p2[i]) for i in range(len(p2))}
        im_sum = raw1["Immature"] + raw1["Mature"]
        if im_sum <= 0:
            p_i = p_m = 0.5
        else:
            p_i, p_m = raw1["Immature"] / im_sum, raw1["Mature"] / im_sum
        q_sum = raw2["Primary"] + raw2["Secondary"]
        if q_sum <= 0:
            q_p = q_s = 0.5
        else:
            q_p, q_s = raw2["Primary"] / q_sum, raw2["Secondary"] / q_sum
        final = {"E-TLS": p_i, "P-TLS": p_m * q_p, "S-TLS": p_m * q_s}
        top = max(final, key=final.get)
        raw_top1 = max(raw1, key=raw1.get)
        if raw_top1 == "Immature":
            production = "E-TLS"
        elif raw_top1 == "Mature":
            production = "P-TLS" if raw2["Primary"] >= raw2["Secondary"] else "S-TLS"
        else:
            production = "uncertain"
        return raw1, raw2, final, top, production

    def predict_rows(self, images: list[np.ndarray]) -> list[dict]:
        p1, p2 = self.raw_probabilities(images)
        return self._prediction_rows(p1, p2)

    def predict_preprocessed(self, images: list[np.ndarray]) -> list[dict]:
        p1, p2 = self.raw_probabilities_preprocessed(images)
        return self._prediction_rows(p1, p2)

    def _prediction_rows(self, p1: np.ndarray, p2: np.ndarray) -> list[dict]:
        rows = []
        for a, b in zip(p1, p2):
            raw1, raw2, final, top, production = self.derived(a, b)
            entropy, margin = entropy_and_margin(final)
            rows.append(
                {
                    "step1": raw1,
                    "step2": raw2,
                    "final": final,
                    "derived_top1": top,
                    "production_class": production,
                    "entropy_normalized": entropy,
                    "top1_margin": margin,
                }
            )
        return rows

    def gradcam(self, image: np.ndarray, model_number: int, target_name: str) -> np.ndarray:
        net = self.net1 if model_number == 1 else self.net2
        names = self.names1 if model_number == 1 else self.names2
        inverse = {v: k for k, v in names.items()}
        target_index = inverse[target_name]
        activation_box: dict[str, torch.Tensor] = {}
        logits_box: dict[str, torch.Tensor] = {}
        head = net.model[-1]

        def activation_hook(_module, _inputs, output):
            activation_box["value"] = output
            output.retain_grad()

        def linear_hook(_module, _inputs, output):
            logits_box["value"] = output

        h1 = head.conv.register_forward_hook(activation_hook)
        h2 = head.linear.register_forward_hook(linear_hook)
        try:
            net.zero_grad(set_to_none=True)
            # Released Ultralytics checkpoints load with frozen parameters.
            # Grad-CAM only needs a differentiable forward graph; enabling
            # gradients on the input does not modify or unfreeze the weights.
            tensor = self.tensors([image], model_number).requires_grad_(True)
            with torch.enable_grad():
                _ = net(tensor)
                score = logits_box["value"][0, target_index]
                score.backward()
                activation = activation_box["value"].detach()
                gradient = activation_box["value"].grad.detach()
                weights = gradient.mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
                cam = torch.nn.functional.interpolate(
                    cam, size=(640, 640), mode="bilinear", align_corners=False
                )[0, 0]
                cam -= cam.min()
                if float(cam.max()) > 0:
                    cam /= cam.max()
                return cam.float().cpu().numpy()
        finally:
            h1.remove()
            h2.remove()


def resize_mask(mask: np.ndarray) -> np.ndarray:
    # All extracted crops are square except boundary clamping; this matches
    # torchvision Resize(shorter edge=640) followed by CenterCrop(640).
    height, width = mask.shape
    scale = 640.0 / min(height, width)
    new_w, new_h = round(width * scale), round(height * scale)
    resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    left = max(0, (new_w - 640) // 2)
    top = max(0, (new_h - 640) // 2)
    return resized[top : top + 640, left : left + 640] > 0


def preprocess_display(rgb: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    scale = 640.0 / min(height, width)
    new_w, new_h = round(width * scale), round(height * scale)
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    left = max(0, (new_w - 640) // 2)
    top = max(0, (new_h - 640) // 2)
    return resized[top : top + 640, left : left + 640]


def overlay_heatmap(rgb640: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    heat = cv2.applyColorMap(np.uint8(np.clip(heatmap, 0, 1) * 255), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.uint8(0.55 * rgb640 + 0.45 * heat)


def saliency_metrics(heatmap: np.ndarray, mask640: np.ndarray) -> dict[str, float]:
    positive = np.clip(heatmap, 0, None)
    total = float(positive.sum())
    inside = float(positive[mask640].sum()) if mask640.any() else 0.0
    area_fraction = float(mask640.mean())
    saliency_fraction = inside / total if total > 0 else float("nan")
    return {
        "polygon_area_fraction": area_fraction,
        "inside_polygon_saliency_fraction": saliency_fraction,
        "inside_polygon_enrichment": (
            saliency_fraction / area_fraction
            if area_fraction > 0 and np.isfinite(saliency_fraction)
            else float("nan")
        ),
        "inside_polygon_mean": float(positive[mask640].mean()) if mask640.any() else float("nan"),
        "outside_polygon_mean": float(positive[~mask640].mean()) if (~mask640).any() else float("nan"),
    }


def occlusion_sensitivity(
    models: CascadeModels,
    baseline_rgb: np.ndarray,
    baseline_mask: np.ndarray,
    window: int = 80,
    stride: int = 40,
) -> tuple[np.ndarray, dict, dict]:
    base640 = models.preprocessed_rgb(baseline_rgb)
    mask640 = resize_mask(baseline_mask)
    baseline = models.predict_preprocessed([base640])[0]
    target = baseline["derived_top1"]
    target_probability = baseline["final"][target]
    blurred = cv2.GaussianBlur(base640, (0, 0), sigmaX=20, sigmaY=20)
    variants, locations = [], []
    positions = list(range(0, 640 - window + 1, stride))
    if positions[-1] != 640 - window:
        positions.append(640 - window)
    for y in positions:
        for x in positions:
            occluded = base640.copy()
            occluded[y : y + window, x : x + window] = blurred[
                y : y + window, x : x + window
            ]
            variants.append(occluded)
            locations.append((x, y))
    predictions = models.predict_preprocessed(variants)
    value = np.zeros((640, 640), dtype=np.float64)
    count = np.zeros((640, 640), dtype=np.float64)
    flip_count = 0
    for pred, (x, y) in zip(predictions, locations):
        drop = target_probability - pred["final"][target]
        value[y : y + window, x : x + window] += drop
        count[y : y + window, x : x + window] += 1
        flip_count += int(pred["derived_top1"] != target)
    heatmap = value / np.maximum(count, 1)
    positive = np.clip(heatmap, 0, None)
    saliency = saliency_metrics(positive, mask640)
    metrics = {
        **saliency,
        "target_class": target,
        "baseline_target_probability": float(target_probability),
        "max_probability_drop": float(heatmap.max()),
        "min_probability_drop": float(heatmap.min()),
        "occlusion_flip_fraction": flip_count / len(predictions),
        "n_occlusions": len(predictions),
        "window_px": window,
        "stride_px": stride,
    }
    return heatmap, metrics, baseline


def counterfactual_images(rgb: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    sigma = max(8.0, min(rgb.shape[:2]) / 25.0)
    blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=sigma, sigmaY=sigma)
    feather_sigma = max(1.5, min(rgb.shape[:2]) / 300.0)
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), feather_sigma)
    alpha = alpha[:, :, None]
    tls_only = np.clip(alpha * rgb + (1.0 - alpha) * blurred, 0, 255).astype(np.uint8)
    context_only = np.clip((1.0 - alpha) * rgb + alpha * blurred, 0, 255).astype(np.uint8)
    return {"original": rgb, "tls_only": tls_only, "context_only": context_only}


def flatten_prediction(pred: dict) -> dict:
    row = {}
    for name, value in pred["step1"].items():
        row[f"step1_p_{name}"] = value
    for name, value in pred["step2"].items():
        row[f"step2_p_{name}"] = value
    for name, value in pred["final"].items():
        row[f"p_{name.replace('-TLS', '')}"] = value
    row.update(
        {
            "derived_top1": pred["derived_top1"],
            "production_class": pred["production_class"],
            "entropy_normalized": pred["entropy_normalized"],
            "top1_margin": pred["top1_margin"],
        }
    )
    return row


def inventory_objects(inventory_path: Path) -> list[dict]:
    """Load an explicit, local object inventory without assuming a lab filesystem.

    The inventory is deliberately excluded from this repository because it may
    contain object coordinates and local WSI locations.  It may be either a JSON
    list or an object with an ``objects`` list.  Relative WSI paths are resolved
    from the inventory file's parent directory.
    """

    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    rows = payload["objects"] if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("Inventory must contain a non-empty JSON object list")

    required = {
        "cohort",
        "sample_id",
        "object_id",
        "wsi",
        "polygon_coordinates",
        "level0_bbox",
        "mpp_x",
        "mpp_y",
        "existing_records",
    }
    objects = []
    for index, row in enumerate(rows):
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(f"Inventory row {index} is missing: {', '.join(missing)}")
        wsi = Path(row["wsi"])
        if not wsi.is_absolute():
            wsi = inventory_path.parent / wsi
        if not wsi.is_file():
            raise FileNotFoundError(f"Inventory row {index} WSI does not exist")
        records = row["existing_records"]
        if not isinstance(records, list) or not records:
            raise ValueError(
                f"Inventory row {index} needs at least one reference crop record"
            )
        reference_paddings = sorted(
            round(float(record["padding_ratio"]), 2) for record in records
        )
        if reference_paddings != [0.10, 0.25]:
            raise ValueError(
                f"Inventory row {index} must contain the 10% and 25% reference crops"
            )
        if any(not record.get("sha256") for record in records):
            raise ValueError(f"Inventory row {index} has a reference crop without SHA-256")
        objects.append(
            {
                "cohort": str(row["cohort"]),
                "sample_id": str(row["sample_id"]),
                "object_id": row["object_id"],
                "wsi": wsi,
                "polygon_coordinates": row["polygon_coordinates"],
                "level0_bbox": row["level0_bbox"],
                "mpp_x": float(row["mpp_x"]),
                "mpp_y": float(row["mpp_y"]),
                "existing_records": records,
            }
        )
    return objects


def write_table(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def create_provenance(
    output: Path,
    args: argparse.Namespace,
    models: CascadeModels,
    reference_path: Path,
    objects: list[dict],
) -> None:
    manifest = {
        "status": "running",
        "scope": "interpretability and robustness; no changes to primary model outputs",
        "n_objects": len(objects),
        "n_slides": len({x["sample_id"] for x in objects}),
        "parameters": {
            "paddings": PADDINGS,
            "occlusion_window_px": args.occlusion_window,
            "occlusion_stride_px": args.occlusion_stride,
            "counterfactual": "Gaussian-blurred removal with feathered polygon boundary",
            "derived_probability_rule": (
                "pE=pI/(pI+pM); pP=pM/(pI+pM)*qP; pS=pM/(pI+pM)*qS"
            ),
        },
        "inputs": {
            "object_inventory": {
                "file": args.inventory.name,
                "sha256": sha256_file(args.inventory),
            },
            "step1": {
                "file": args.step1.name,
                "sha256": sha256_file(args.step1),
                "names": models.names1,
            },
            "step2": {
                "file": args.step2.name,
                "sha256": sha256_file(args.step2),
                "names": models.names2,
            },
            "stain_reference": {
                "file": reference_path.name,
                "sha256": sha256_file(reference_path),
            },
            "script": {
                "file": Path(__file__).name,
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": __import__("ultralytics").__version__,
            "opencv": cv2.__version__,
            "pillow": PIL.__version__,
            "cuda": torch.version.cuda,
            "device": str(models.device),
        },
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory",
        required=True,
        type=Path,
        help="Local JSON object inventory; never copied into the output",
    )
    parser.add_argument("--step1", required=True, type=Path)
    parser.add_argument("--step2", required=True, type=Path)
    parser.add_argument("--stain-reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--occlusion-window", type=int, default=80)
    parser.add_argument("--occlusion-stride", type=int, default=40)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test first N objects")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists: {args.output}")
        if any(args.output.iterdir()):
            raise SystemExit("Refusing to overwrite a non-empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    for sub in ["gradcam", "occlusion", "counterfactual", "padding_curves"]:
        (args.output / sub).mkdir()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise SystemExit("CUDA requested but unavailable")

    objects = inventory_objects(args.inventory)
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be positive")
        objects = objects[: args.limit]
    models = CascadeModels(args.step1, args.step2, args.device)
    reference_rgb = np.asarray(Image.open(args.stain_reference).convert("RGB"))
    reinhard_ref = reinhard_reference(reference_rgb)
    macenko_ref = macenko_reference(reference_rgb)
    create_provenance(args.output, args, models, args.stain_reference, objects)

    padding_rows, stain_rows, counterfactual_rows = [], [], []
    gradcam_rows, occlusion_rows, crop_validation_rows = [], [], []
    slide_cache: dict[str, openslide.OpenSlide] = {}

    try:
        for object_index, obj in enumerate(objects, 1):
            sample_id = obj["sample_id"]
            object_id = obj["object_id"]
            prefix = f"{sample_id}_obj{object_id}"
            print(f"[{object_index}/{len(objects)}] {prefix}", flush=True)
            if sample_id not in slide_cache:
                slide_cache[sample_id] = openslide.OpenSlide(str(obj["wsi"]))
            slide = slide_cache[sample_id]

            crops: dict[float, tuple[np.ndarray, np.ndarray, dict]] = {}
            encoded95: dict[float, np.ndarray] = {}
            for padding in PADDINGS:
                rgb, mask, geometry = crop_object(slide, obj, padding)
                jpeg95 = encode_roundtrip(rgb, "JPEG", 95)
                crops[padding] = (rgb, mask, geometry)
                encoded95[padding] = jpeg95

            # Gate arbitrary recropping against the original 10% and 25% JPEGs.
            for record in obj["existing_records"]:
                padding = round(float(record["padding_ratio"]), 2)
                if padding not in encoded95:
                    continue
                buffer = io.BytesIO()
                Image.fromarray(crops[padding][0]).save(buffer, format="JPEG", quality=95)
                generated_hash = hashlib.sha256(buffer.getvalue()).hexdigest()
                crop_validation_rows.append(
                    {
                        "cohort": obj["cohort"],
                        "sample_id": sample_id,
                        "object_id": object_id,
                        "padding": padding,
                        "expected_sha256": record["sha256"],
                        "generated_sha256": generated_hash,
                        "exact_match": generated_hash == record["sha256"],
                    }
                )

            padding_predictions = models.predict_rows([encoded95[x] for x in PADDINGS])
            for padding, prediction in zip(PADDINGS, padding_predictions):
                row = {
                    "cohort": obj["cohort"],
                    "sample_id": sample_id,
                    "object_id": object_id,
                    "padding": padding,
                    **crops[padding][2],
                    **flatten_prediction(prediction),
                }
                padding_rows.append(row)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
            for cls, color in zip(FINAL_CLASSES, ["#F2CF5B", "#4C78A8", "#E15759"]):
                values = [x["final"][cls] for x in padding_predictions]
                axes[0].plot(np.array(PADDINGS) * 100, values, "-o", label=cls, color=color)
            axes[0].set_ylim(-0.03, 1.03)
            axes[0].set_xlabel("Padding (%)")
            axes[0].set_ylabel("Derived hierarchical score")
            axes[0].set_title("E/P/S curve")
            axes[0].legend()
            axes[1].plot(
                np.array(PADDINGS) * 100,
                [x["entropy_normalized"] for x in padding_predictions],
                "-o",
                label="Normalized entropy",
                color="#9467BD",
            )
            axes[1].plot(
                np.array(PADDINGS) * 100,
                [x["top1_margin"] for x in padding_predictions],
                "-o",
                label="Top1 margin",
                color="#59A14F",
            )
            axes[1].set_ylim(-0.03, 1.03)
            axes[1].set_xlabel("Padding (%)")
            axes[1].set_title("Uncertainty and separation")
            axes[1].legend()
            fig.suptitle(prefix)
            fig.savefig(args.output / "padding_curves" / f"{prefix}.png", dpi=150)
            plt.close(fig)

            # Stain, colour and encoding sensitivity at 10% padding.
            rgb10, _, _ = crops[0.10]
            variants = {
                "raw_rgb": rgb10,
                "png_lossless": encode_roundtrip(rgb10, "PNG"),
                "jpeg95": encode_roundtrip(rgb10, "JPEG", 95),
                "jpeg90": encode_roundtrip(rgb10, "JPEG", 90),
                "jpeg80": encode_roundtrip(rgb10, "JPEG", 80),
                "reinhard": reinhard_normalize(rgb10, reinhard_ref),
                "macenko": macenko_normalize(rgb10, macenko_ref),
                "hue_minus_0.02": hue_shift(rgb10, -0.02),
                "hue_plus_0.02": hue_shift(rgb10, 0.02),
                "saturation_0.90": saturation_scale(rgb10, 0.90),
                "saturation_1.10": saturation_scale(rgb10, 1.10),
                "brightness_0.90": brightness_scale(rgb10, 0.90),
                "brightness_1.10": brightness_scale(rgb10, 1.10),
            }
            variant_predictions = models.predict_rows(list(variants.values()))
            baseline_variant = variant_predictions[0]
            for variant_name, prediction in zip(variants, variant_predictions):
                stain_rows.append(
                    {
                        "cohort": obj["cohort"],
                        "sample_id": sample_id,
                        "object_id": object_id,
                        "variant": variant_name,
                        "baseline_top1": baseline_variant["derived_top1"],
                        "class_flip_vs_raw": prediction["derived_top1"]
                        != baseline_variant["derived_top1"],
                        "production_flip_vs_raw": prediction["production_class"]
                        != baseline_variant["production_class"],
                        **flatten_prediction(prediction),
                    }
                )

            # Counterfactuals at both production padding values.
            for padding in [0.10, 0.25]:
                rgb, mask, _ = crops[padding]
                counterfactuals = counterfactual_images(rgb, mask)
                predictions = models.predict_rows(list(counterfactuals.values()))
                baseline_cf = predictions[0]
                fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
                for ax, (variant_name, image), prediction in zip(
                    axes, counterfactuals.items(), predictions
                ):
                    ax.imshow(image)
                    ax.axis("off")
                    ax.set_title(
                        f"{variant_name}\n{prediction['derived_top1']} "
                        f"p={prediction['final'][prediction['derived_top1']]:.3f}"
                    )
                    counterfactual_rows.append(
                        {
                            "cohort": obj["cohort"],
                            "sample_id": sample_id,
                            "object_id": object_id,
                            "padding": padding,
                            "variant": variant_name,
                            "baseline_top1": baseline_cf["derived_top1"],
                            "class_flip_vs_original": prediction["derived_top1"]
                            != baseline_cf["derived_top1"],
                            "baseline_target_probability": baseline_cf["final"][
                                baseline_cf["derived_top1"]
                            ],
                            "baseline_target_probability_after_counterfactual": prediction[
                                "final"
                            ][baseline_cf["derived_top1"]],
                            "baseline_target_probability_drop": baseline_cf["final"][
                                baseline_cf["derived_top1"]
                            ]
                            - prediction["final"][baseline_cf["derived_top1"]],
                            **flatten_prediction(prediction),
                        }
                    )
                fig.suptitle(f"{prefix}, padding={int(padding * 100)}%")
                fig.savefig(
                    args.output / "counterfactual" / f"{prefix}_pad{int(padding * 100)}.png",
                    dpi=140,
                )
                plt.close(fig)

            # Grad-CAM at 10% production padding, using exact JPEG95 pixels.
            base_rgb = encoded95[0.10]
            base_mask = crops[0.10][1]
            display = np.uint8(
                np.clip(models.preprocessed_rgb(base_rgb) * 255.0, 0, 255)
            )
            mask640 = resize_mask(base_mask)
            baseline = models.predict_rows([base_rgb])[0]
            step1_target = (
                "Immature" if baseline["derived_top1"] == "E-TLS" else "Mature"
            )
            step2_target = max(baseline["step2"], key=baseline["step2"].get)
            cam1 = models.gradcam(base_rgb, 1, step1_target)
            cam2 = models.gradcam(base_rgb, 2, step2_target)
            for model_name, target, cam in [
                ("step1", step1_target, cam1),
                ("step2", step2_target, cam2),
            ]:
                gradcam_rows.append(
                    {
                        "cohort": obj["cohort"],
                        "sample_id": sample_id,
                        "object_id": object_id,
                        "model": model_name,
                        "target_class": target,
                        **saliency_metrics(cam, mask640),
                    }
                )
            fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
            axes[0].imshow(display)
            axes[0].contour(mask640, levels=[0.5], colors=["lime"], linewidths=1)
            axes[0].set_title(f"Input\n{baseline['derived_top1']}")
            axes[1].imshow(overlay_heatmap(display, cam1))
            axes[1].contour(mask640, levels=[0.5], colors=["lime"], linewidths=1)
            axes[1].set_title(f"Step 1 Grad-CAM: {step1_target}")
            axes[2].imshow(overlay_heatmap(display, cam2))
            axes[2].contour(mask640, levels=[0.5], colors=["lime"], linewidths=1)
            axes[2].set_title(f"Step 2 Grad-CAM: {step2_target}")
            for ax in axes:
                ax.axis("off")
            fig.suptitle(prefix)
            fig.savefig(args.output / "gradcam" / f"{prefix}.png", dpi=150)
            plt.close(fig)

            # Occlusion sensitivity at 10% padding.
            occ, occ_metrics, occ_baseline = occlusion_sensitivity(
                models,
                base_rgb,
                base_mask,
                window=args.occlusion_window,
                stride=args.occlusion_stride,
            )
            occlusion_rows.append(
                {
                    "cohort": obj["cohort"],
                    "sample_id": sample_id,
                    "object_id": object_id,
                    **occ_metrics,
                }
            )
            max_abs = max(abs(float(occ.min())), abs(float(occ.max())), 1e-6)
            fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
            axes[0].imshow(display)
            axes[0].contour(mask640, levels=[0.5], colors=["lime"], linewidths=1)
            axes[0].set_title(
                f"Input: {occ_baseline['derived_top1']}\n"
                f"p={occ_baseline['final'][occ_baseline['derived_top1']]:.3f}"
            )
            image = axes[1].imshow(occ, cmap="coolwarm", vmin=-max_abs, vmax=max_abs)
            axes[1].contour(mask640, levels=[0.5], colors=["lime"], linewidths=1)
            axes[1].set_title("Target probability drop")
            fig.colorbar(image, ax=axes[1], fraction=0.046)
            axes[2].imshow(overlay_heatmap(display, np.clip(occ, 0, None) / max(occ.max(), 1e-6)))
            axes[2].contour(mask640, levels=[0.5], colors=["lime"], linewidths=1)
            axes[2].set_title("Positive occlusion saliency")
            for ax in axes:
                ax.axis("off")
            fig.suptitle(prefix)
            fig.savefig(args.output / "occlusion" / f"{prefix}.png", dpi=150)
            plt.close(fig)
    finally:
        for slide in slide_cache.values():
            slide.close()

    write_table(args.output / "crop_reconstruction_validation.tsv", crop_validation_rows)
    write_table(args.output / "padding_curves.tsv", padding_rows)
    write_table(args.output / "stain_encoding_sensitivity.tsv", stain_rows)
    write_table(args.output / "counterfactual_sensitivity.tsv", counterfactual_rows)
    write_table(args.output / "gradcam_metrics.tsv", gradcam_rows)
    write_table(args.output / "occlusion_metrics.tsv", occlusion_rows)

    validation = pd.DataFrame(crop_validation_rows)
    if not bool(validation["exact_match"].all()):
        bad = validation.loc[~validation["exact_match"]]
        raise RuntimeError(
            f"Crop reconstruction did not match {len(bad)} existing JPEG hashes; results retained for audit"
        )

    manifest_path = args.output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "complete"
    manifest["outputs"] = {}
    for path in sorted(args.output.glob("*.tsv")):
        manifest["outputs"][path.name] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    with (args.output / "SHA256SUMS").open("w") as handle:
        for path in sorted(args.output.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                handle.write(f"{sha256_file(path)}  {path.relative_to(args.output)}\n")
    print(f"INTERPRETABILITY_SUCCESS output={args.output}")


if __name__ == "__main__":
    main()
