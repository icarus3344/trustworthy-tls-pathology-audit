#!/usr/bin/env python3
"""Summarize a YOLO TLS interpretability and robustness audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

KEY = ["cohort", "sample_id", "object_id"]
CLASSES = ["E-TLS", "P-TLS", "S-TLS"]
COLORS = {"E-TLS": "#F2CF5B", "P-TLS": "#4C78A8", "S-TLS": "#E15759"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().eq("true")


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=14,
        va="top",
    )


def padding_summary(padding: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    padding = padding.sort_values(KEY + ["padding"]).copy()
    object_rows = []
    for keys, group in padding.groupby(KEY, sort=False):
        group = group.sort_values("padding")
        classes = group["derived_top1"].tolist()
        base = group.loc[np.isclose(group["padding"], 0.10)].iloc[0]
        transitions = sum(a != b for a, b in zip(classes[:-1], classes[1:]))
        changed = group.loc[group["derived_top1"] != base["derived_top1"], "padding"]
        object_rows.append(
            {
                **dict(zip(KEY, keys)),
                "baseline_class_pad10": base["derived_top1"],
                "baseline_production_class_pad10": base["production_class"],
                "padding_stable_all": len(set(classes)) == 1,
                "padding_n_transitions": transitions,
                "padding_first_difference_from_10": (
                    float(changed.min()) if len(changed) else np.nan
                ),
                "padding_min_margin": float(group["top1_margin"].min()),
                "padding_max_entropy": float(group["entropy_normalized"].max()),
                "padding_pE_range": float(group["p_E"].max() - group["p_E"].min()),
                "padding_pP_range": float(group["p_P"].max() - group["p_P"].min()),
                "padding_pS_range": float(group["p_S"].max() - group["p_S"].min()),
            }
        )
    objects = pd.DataFrame(object_rows)

    base = padding.loc[np.isclose(padding["padding"], 0.10), KEY + ["derived_top1"]].rename(
        columns={"derived_top1": "baseline_class"}
    )
    comparison = padding.merge(base, on=KEY, validate="many_to_one")
    comparison["flip_vs_pad10"] = comparison["derived_top1"] != comparison["baseline_class"]
    by_padding = comparison.groupby("padding", as_index=False).agg(
        n_objects=("object_id", "size"),
        flip_fraction=("flip_vs_pad10", "mean"),
        mean_entropy=("entropy_normalized", "mean"),
        mean_margin=("top1_margin", "mean"),
        mean_pE=("p_E", "mean"),
        mean_pP=("p_P", "mean"),
        mean_pS=("p_S", "mean"),
    )
    return objects, by_padding


def stain_summary(stain: pd.DataFrame) -> pd.DataFrame:
    stain = stain.copy()
    stain["class_flip_vs_raw"] = bool_series(stain["class_flip_vs_raw"])
    stain["production_flip_vs_raw"] = bool_series(stain["production_flip_vs_raw"])
    stain["ps_flip_vs_raw"] = (
        stain["baseline_top1"].isin(["P-TLS", "S-TLS"])
        & stain["derived_top1"].isin(["P-TLS", "S-TLS"])
        & (stain["baseline_top1"] != stain["derived_top1"])
    )
    raw = stain.loc[stain["variant"] == "raw_rgb", KEY + ["p_E", "p_P", "p_S"]].rename(
        columns={"p_E": "raw_p_E", "p_P": "raw_p_P", "p_S": "raw_p_S"}
    )
    stain = stain.merge(raw, on=KEY, validate="many_to_one")
    stain["max_abs_probability_change"] = (
        stain[["p_E", "p_P", "p_S"]]
        .sub(stain[["raw_p_E", "raw_p_P", "raw_p_S"]].to_numpy())
        .abs()
        .max(axis=1)
    )
    summary = stain.groupby("variant", as_index=False).agg(
        n_objects=("object_id", "size"),
        class_flip_fraction=("class_flip_vs_raw", "mean"),
        production_flip_fraction=("production_flip_vs_raw", "mean"),
        ps_flip_fraction=("ps_flip_vs_raw", "mean"),
        mean_max_abs_probability_change=("max_abs_probability_change", "mean"),
        median_entropy=("entropy_normalized", "median"),
        median_margin=("top1_margin", "median"),
    )
    return summary


def counterfactual_summary(counter: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counter = counter.copy()
    counter["class_flip_vs_original"] = bool_series(counter["class_flip_vs_original"])
    summary = counter.groupby(["padding", "variant"], as_index=False).agg(
        n_objects=("object_id", "size"),
        class_flip_fraction=("class_flip_vs_original", "mean"),
        mean_target_probability_drop=("baseline_target_probability_drop", "mean"),
        median_target_probability_drop=("baseline_target_probability_drop", "median"),
    )
    pivot = counter.pivot_table(
        index=KEY + ["padding"],
        columns="variant",
        values=["baseline_target_probability_drop", "class_flip_vs_original"],
        aggfunc="first",
    )
    pivot.columns = ["_".join(map(str, x)) for x in pivot.columns]
    pivot = pivot.reset_index()
    pivot["attribution_tendency"] = np.select(
        [
            pivot["baseline_target_probability_drop_context_only"]
            > pivot["baseline_target_probability_drop_tls_only"] + 0.05,
            pivot["baseline_target_probability_drop_tls_only"]
            > pivot["baseline_target_probability_drop_context_only"] + 0.05,
        ],
        ["TLS_morphology_dominant", "context_dominant"],
        default="mixed_or_small_effect",
    )
    return summary, pivot


def integrated_objects(
    padding_objects: pd.DataFrame,
    stain: pd.DataFrame,
    counter_pivot: pd.DataFrame,
    gradcam: pd.DataFrame,
    occlusion: pd.DataFrame,
) -> pd.DataFrame:
    stain_nontrivial = stain.loc[stain["variant"] != "raw_rgb"].copy()
    stain_nontrivial["class_flip_vs_raw"] = bool_series(stain_nontrivial["class_flip_vs_raw"])
    stain_obj = stain_nontrivial.groupby(KEY, as_index=False).agg(
        any_stain_or_encoding_flip=("class_flip_vs_raw", "max"),
        n_stain_or_encoding_flips=("class_flip_vs_raw", "sum"),
    )
    counter_obj = counter_pivot.groupby(KEY, as_index=False).agg(
        mean_tls_removal_drop=("baseline_target_probability_drop_context_only", "mean"),
        mean_context_removal_drop=("baseline_target_probability_drop_tls_only", "mean"),
        any_tls_removal_flip=("class_flip_vs_original_context_only", "max"),
        any_context_removal_flip=("class_flip_vs_original_tls_only", "max"),
    )
    cam = gradcam.pivot_table(
        index=KEY,
        columns="model",
        values="inside_polygon_enrichment",
        aggfunc="first",
    ).reset_index()
    cam = cam.rename(
        columns={
            "step1": "gradcam_step1_polygon_enrichment",
            "step2": "gradcam_step2_polygon_enrichment",
        }
    )
    occ = occlusion[
        KEY
        + [
            "inside_polygon_enrichment",
            "occlusion_flip_fraction",
            "max_probability_drop",
        ]
    ].rename(
        columns={
            "inside_polygon_enrichment": "occlusion_polygon_enrichment",
        }
    )
    result = padding_objects.merge(stain_obj, on=KEY, validate="one_to_one")
    result = result.merge(counter_obj, on=KEY, validate="one_to_one")
    result = result.merge(cam, on=KEY, validate="one_to_one")
    result = result.merge(occ, on=KEY, validate="one_to_one")

    risk = (
        (~result["padding_stable_all"]).astype(int)
        + result["any_stain_or_encoding_flip"].astype(int)
        + result["any_context_removal_flip"].astype(int)
        + (result["gradcam_step2_polygon_enrichment"] < 1.0).astype(int)
        + (
            result["occlusion_polygon_enrichment"].isna()
            | (result["occlusion_polygon_enrichment"] < 1.0)
        ).astype(int)
        + (result["padding_min_margin"] < 0.20).astype(int)
    )
    result["engineering_risk_score"] = risk
    result["engineering_review_priority"] = pd.cut(
        risk,
        bins=[-1, 0, 2, 99],
        labels=["low", "medium", "high"],
    ).astype(str)
    return result


def plot_padding(padding: pd.DataFrame, by_padding: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    x = by_padding["padding"] * 100
    for cls, column in zip(CLASSES, ["mean_pE", "mean_pP", "mean_pS"]):
        axes[0, 0].plot(x, by_padding[column], "-o", color=COLORS[cls], label=cls)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_xlabel("Padding (%)")
    axes[0, 0].set_ylabel("Mean derived score")
    axes[0, 0].set_title("Mean E/P/S curve")
    axes[0, 0].legend()

    class_counts = (
        padding.groupby(["padding", "derived_top1"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CLASSES, fill_value=0)
    )
    proportions = class_counts.div(class_counts.sum(axis=1), axis=0)
    axes[0, 1].stackplot(
        proportions.index * 100,
        *[proportions[c] for c in CLASSES],
        labels=CLASSES,
        colors=[COLORS[x] for x in CLASSES],
        alpha=0.85,
    )
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_xlabel("Padding (%)")
    axes[0, 1].set_ylabel("Class proportion")
    axes[0, 1].set_title("Derived top-1 composition")
    axes[0, 1].legend(loc="upper right")

    axes[1, 0].plot(x, by_padding["mean_entropy"], "-o", label="Entropy", color="#9467BD")
    axes[1, 0].plot(x, by_padding["mean_margin"], "-o", label="Top1 margin", color="#59A14F")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_xlabel("Padding (%)")
    axes[1, 0].set_title("Mean uncertainty")
    axes[1, 0].legend()

    axes[1, 1].bar(x.astype(str), by_padding["flip_fraction"], color="#E15759")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xlabel("Padding (%)")
    axes[1, 1].set_ylabel("Fraction differing from 10%")
    axes[1, 1].set_title("Class sensitivity relative to production 10% crop")
    for label, ax in zip("ABCD", axes.flat):
        add_panel_label(ax, label)
    fig.suptitle("TLS maturity sensitivity across polygon padding")
    fig.savefig(output / "figure1_multiscale_padding.png", dpi=220)
    fig.savefig(output / "figure1_multiscale_padding.pdf")
    plt.close(fig)


def plot_stain(summary: pd.DataFrame, output: Path) -> None:
    order = [
        "png_lossless",
        "jpeg95",
        "jpeg90",
        "jpeg80",
        "reinhard",
        "macenko",
        "hue_minus_0.02",
        "hue_plus_0.02",
        "saturation_0.90",
        "saturation_1.10",
        "brightness_0.90",
        "brightness_1.10",
    ]
    x = summary.set_index("variant").reindex(order).reset_index()
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    pos = np.arange(len(x))
    axes[0].bar(pos - 0.18, x["class_flip_fraction"], 0.36, label="Any E/P/S flip")
    axes[0].bar(pos + 0.18, x["ps_flip_fraction"], 0.36, label="P↔S flip")
    axes[0].set_ylim(
        0, max(0.05, float(x[["class_flip_fraction", "ps_flip_fraction"]].max().max()) * 1.25)
    )
    axes[0].set_xticks(pos, x["variant"], rotation=35, ha="right")
    axes[0].set_ylabel("Object fraction")
    axes[0].set_title("Class-flip sensitivity relative to raw RGB")
    axes[0].legend()
    axes[1].bar(pos, x["mean_max_abs_probability_change"], color="#4C78A8")
    axes[1].set_xticks(pos, x["variant"], rotation=35, ha="right")
    axes[1].set_ylabel("Mean max |Δ probability|")
    axes[1].set_title("Probability sensitivity without requiring a hard flip")
    add_panel_label(axes[0], "A")
    add_panel_label(axes[1], "B")
    fig.suptitle("Stain, colour and encoding robustness")
    fig.savefig(output / "figure2_stain_encoding.png", dpi=220)
    fig.savefig(output / "figure2_stain_encoding.pdf")
    plt.close(fig)


def plot_counterfactual(
    counter: pd.DataFrame, summary: pd.DataFrame, pivot: pd.DataFrame, output: Path
) -> None:
    variants = ["tls_only", "context_only"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for i, padding in enumerate([0.10, 0.25]):
        subset = summary.loc[
            np.isclose(summary["padding"], padding) & summary["variant"].isin(variants)
        ].set_index("variant")
        axes[0, i].bar(
            ["Context removed\n(TLS-only)", "TLS removed\n(context-only)"],
            [
                subset.loc["tls_only", "class_flip_fraction"],
                subset.loc["context_only", "class_flip_fraction"],
            ],
            color=["#F28E2B", "#4C78A8"],
        )
        axes[0, i].set_ylim(0, 1)
        axes[0, i].set_ylabel("Class-flip fraction")
        axes[0, i].set_title(f"Padding {int(padding * 100)}%")

    axes[1, 0].scatter(
        pivot["baseline_target_probability_drop_tls_only"],
        pivot["baseline_target_probability_drop_context_only"],
        c=pivot["padding"].map({0.10: "#4C78A8", 0.25: "#E15759"}),
        alpha=0.7,
    )
    lo = min(
        pivot["baseline_target_probability_drop_tls_only"].min(),
        pivot["baseline_target_probability_drop_context_only"].min(),
    )
    hi = max(
        pivot["baseline_target_probability_drop_tls_only"].max(),
        pivot["baseline_target_probability_drop_context_only"].max(),
    )
    axes[1, 0].plot([lo, hi], [lo, hi], color="#555555", linewidth=0.8)
    axes[1, 0].set_xlabel("Target-probability drop when context is removed")
    axes[1, 0].set_ylabel("Target-probability drop when TLS is removed")
    axes[1, 0].set_title("TLS morphology versus context dependence")

    counts = (
        pivot["attribution_tendency"]
        .value_counts()
        .reindex(
            ["TLS_morphology_dominant", "context_dominant", "mixed_or_small_effect"], fill_value=0
        )
    )
    axes[1, 1].barh(
        ["TLS morphology dominant", "Context dominant", "Mixed/small effect"],
        counts,
        color=["#4C78A8", "#F28E2B", "#BAB0AC"],
    )
    axes[1, 1].set_xlabel("Object × padding observations")
    axes[1, 1].set_title("Counterfactual attribution tendency")
    for label, ax in zip("ABCD", axes.flat):
        add_panel_label(ax, label)
    fig.suptitle("TLS-only and context-only counterfactual audit")
    fig.savefig(output / "figure3_counterfactual.png", dpi=220)
    fig.savefig(output / "figure3_counterfactual.pdf")
    plt.close(fig)


def plot_saliency(
    gradcam: pd.DataFrame, occlusion: pd.DataFrame, objects: pd.DataFrame, output: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    gradcam.boxplot(
        column="inside_polygon_enrichment",
        by="model",
        ax=axes[0, 0],
        grid=False,
    )
    axes[0, 0].axhline(1, color="#555555", linewidth=0.8)
    axes[0, 0].set_title("Grad-CAM enrichment within TLS polygon")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("Saliency enrichment over polygon area")

    axes[0, 1].hist(occlusion["inside_polygon_enrichment"].dropna(), bins=20, color="#4C78A8")
    axes[0, 1].axvline(1, color="#555555", linewidth=0.8)
    axes[0, 1].set_xlabel("Occlusion saliency enrichment")
    axes[0, 1].set_ylabel("Objects")
    axes[0, 1].set_title("Occlusion evidence inside polygon")

    cam2 = gradcam.loc[gradcam["model"] == "step2", KEY + ["inside_polygon_enrichment"]].rename(
        columns={"inside_polygon_enrichment": "cam2"}
    )
    joined = cam2.merge(
        occlusion[KEY + ["inside_polygon_enrichment"]],
        on=KEY,
        validate="one_to_one",
    )
    axes[1, 0].scatter(
        joined["cam2"], joined["inside_polygon_enrichment"], alpha=0.65, color="#59A14F"
    )
    axes[1, 0].axhline(1, color="#777777", linewidth=0.7)
    axes[1, 0].axvline(1, color="#777777", linewidth=0.7)
    axes[1, 0].set_xlabel("Step-2 Grad-CAM enrichment")
    axes[1, 0].set_ylabel("Occlusion enrichment")
    axes[1, 0].set_title("Agreement of two attribution methods")

    priorities = (
        objects["engineering_review_priority"]
        .value_counts()
        .reindex(["low", "medium", "high"], fill_value=0)
    )
    axes[1, 1].bar(priorities.index, priorities.values, color=["#59A14F", "#F2CF5B", "#E15759"])
    axes[1, 1].set_ylabel("Objects")
    axes[1, 1].set_title("Integrated engineering review priority")
    for label, ax in zip("ABCD", axes.flat):
        add_panel_label(ax, label)
    fig.suptitle("Grad-CAM and occlusion sensitivity")
    fig.savefig(output / "figure4_saliency.png", dpi=220)
    fig.savefig(output / "figure4_saliency.pdf")
    plt.close(fig)
    plt.close("all")


def build_priority_review_panels(source: Path, objects: pd.DataFrame, output: Path) -> None:
    panel_dir = output / "priority_object_panels"
    sheet_dir = output / "priority_contact_sheets"
    panel_dir.mkdir()
    sheet_dir.mkdir()
    priority_order = {"high": 0, "medium": 1, "low": 2}
    selected = objects.copy()
    selected["_priority_order"] = selected["engineering_review_priority"].map(priority_order)
    selected = selected.sort_values(
        ["_priority_order", "engineering_risk_score", "padding_min_margin"],
        ascending=[True, False, True],
    ).head(24)
    selected.drop(columns="_priority_order").to_csv(
        output / "priority_review_objects.tsv", sep="\t", index=False
    )

    generated = []
    for row in selected.itertuples():
        prefix = f"{row.sample_id}_obj{row.object_id}"
        paths = [
            source / "gradcam" / f"{prefix}.png",
            source / "occlusion" / f"{prefix}.png",
            source / "counterfactual" / f"{prefix}_pad10.png",
            source / "padding_curves" / f"{prefix}.png",
        ]
        if not all(x.exists() for x in paths):
            raise FileNotFoundError(f"Missing review component for {prefix}: {paths}")
        tiles = []
        for path in paths:
            image = Image.open(path).convert("RGB")
            image.thumbnail((1050, 520), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (1060, 530), "white")
            tile.paste(image, ((1060 - image.width) // 2, (530 - image.height) // 2))
            tiles.append(tile)
        canvas = Image.new("RGB", (2120, 1120), "white")
        for i, tile in enumerate(tiles):
            canvas.paste(tile, ((i % 2) * 1060, 60 + (i // 2) * 530))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (20, 18),
            f"{prefix} | priority={row.engineering_review_priority} "
            f"risk={row.engineering_risk_score} | pad10={row.baseline_class_pad10}",
            fill="black",
        )
        panel_path = panel_dir / f"{prefix}.jpg"
        canvas.save(panel_path, quality=92)
        generated.append(panel_path)

    # Source panels are approximately 2:1 landscape composites. Two
    # full-width panels per page keep labels readable during presentation.
    for start in range(0, len(generated), 2):
        page_files = generated[start : start + 2]
        page = Image.new("RGB", (1650, 1750), "white")
        draw = ImageDraw.Draw(page)
        draw.text(
            (25, 18),
            f"YOLO TLS interpretability priority review — page {start // 2 + 1}",
            fill="black",
        )
        for i, path in enumerate(page_files):
            image = Image.open(path).convert("RGB")
            image.thumbnail((1600, 800), Image.Resampling.LANCZOS)
            x = 25 + (1600 - image.width) // 2
            y = 65 + i * 830 + (800 - image.height) // 2
            page.paste(image, (x, y))
        page.save(sheet_dir / f"priority_review_p{start // 2 + 1}.jpg", quality=92)


def markdown_assessment(
    source: Path,
    padding_objects: pd.DataFrame,
    stain_summary_df: pd.DataFrame,
    counter_summary: pd.DataFrame,
    counter_pivot: pd.DataFrame,
    gradcam: pd.DataFrame,
    occlusion: pd.DataFrame,
    objects: pd.DataFrame,
) -> str:
    stable = int(padding_objects["padding_stable_all"].sum())
    n = len(padding_objects)
    stain_idx = stain_summary_df.set_index("variant")
    png_change = stain_idx.loc["png_lossless", "mean_max_abs_probability_change"]
    lines = [
        "# YOLO TLS interpretability and robustness assessment",
        "",
        f"Scope: {n} model-generated objects from "
        f"{objects.sample_id.nunique()} slides with postprocessed candidates.",
        "This is an engineering audit, not a pathology accuracy study; slides with zero",
        "postprocessed candidates are not represented in these object-level tables.",
        "",
        "## Dataset and gates",
        "",
        f"- Slides contributing objects: {objects.sample_id.nunique()}; objects: {n}.",
        "- Reconstructed 10%/25% JPEG crops were required to match the existing SHA-256 exactly.",
        f"- Raw-RGB versus lossless-PNG mean maximum probability difference: {png_change:.3g}.",
        "- Step-1 four-class raw probabilities are retained. E/P/S values are explicitly derived",
        "  hierarchical scores, not calibrated posteriors or an author-confirmed rule.",
        "",
        "## Multiscale padding",
        "",
        f"- Stable from 0% through 40%: {stable}/{n} ({stable / n:.1%}).",
        f"- At least one class transition: {n - stable}/{n} ({1 - stable / n:.1%}).",
        f"- Median minimum top-1 margin: {padding_objects.padding_min_margin.median():.3f}.",
        f"- Median maximum normalized entropy: {padding_objects.padding_max_entropy.median():.3f}.",
        "",
        "## Stain, colour and encoding",
        "",
    ]
    for variant in [
        "jpeg95",
        "jpeg90",
        "jpeg80",
        "reinhard",
        "macenko",
        "hue_minus_0.02",
        "hue_plus_0.02",
        "saturation_0.90",
        "saturation_1.10",
        "brightness_0.90",
        "brightness_1.10",
    ]:
        row = stain_idx.loc[variant]
        lines.append(
            f"- {variant}: any flip {row.class_flip_fraction:.1%}; "
            f"P↔S flip {row.ps_flip_fraction:.1%}; "
            f"mean max |Δp| {row.mean_max_abs_probability_change:.3f}."
        )
    lines += [
        "",
        "## TLS-only/context-only counterfactuals",
        "",
    ]
    for padding in [0.10, 0.25]:
        frame = counter_summary.loc[np.isclose(counter_summary.padding, padding)].set_index(
            "variant"
        )
        lines.append(
            f"- Padding {padding:.0%}: context removal (TLS-only) flipped "
            f"{frame.loc['tls_only', 'class_flip_fraction']:.1%}; TLS removal "
            f"(context-only) flipped {frame.loc['context_only', 'class_flip_fraction']:.1%}."
        )
    tendency = counter_pivot["attribution_tendency"].value_counts()
    lines += [
        f"- TLS-morphology-dominant observations: {int(tendency.get('TLS_morphology_dominant', 0))}.",
        f"- Context-dominant observations: {int(tendency.get('context_dominant', 0))}.",
        f"- Mixed/small-effect observations: {int(tendency.get('mixed_or_small_effect', 0))}.",
        "",
        "Counterfactual images are intentionally manipulated and partly out of distribution;",
        "they identify dependence, not a biological causal mechanism.",
        "",
        "## Grad-CAM and occlusion",
        "",
    ]
    for model in ["step1", "step2"]:
        values = gradcam.loc[gradcam.model == model, "inside_polygon_enrichment"]
        lines.append(
            f"- {model} median polygon saliency enrichment: {values.median():.3f}; "
            f"fraction >1: {(values > 1).mean():.1%}."
        )
    lines += [
        f"- Occlusion median polygon enrichment: {occlusion.inside_polygon_enrichment.median():.3f}; "
        f"fraction >1: {(occlusion.inside_polygon_enrichment > 1).mean():.1%}.",
        f"- Objects with no positive occlusion-probability drop (undefined enrichment): "
        f"{int(occlusion.inside_polygon_enrichment.isna().sum())}.",
        f"- Median occlusion-induced flip fraction: {occlusion.occlusion_flip_fraction.median():.1%}.",
        "",
        "## Integrated interpretation",
        "",
    ]
    priorities = objects.engineering_review_priority.value_counts()
    lines += [
        f"- Low/medium/high engineering review priority: "
        f"{int(priorities.get('low', 0))}/{int(priorities.get('medium', 0))}/{int(priorities.get('high', 0))}.",
        "- A robust object should have a broad padding plateau, small encoding/colour changes,",
        "  Step-2 and occlusion evidence enriched within the polygon, and low dependence on",
        "  context-only visual information.",
        "- None of these tests establishes E/P/S correctness. They test whether the classifier",
        "  behaves consistently and attends to a plausible location.",
        "",
        "## Files",
        "",
        f"- Source audit directory label: `{source.name}`.",
        "- `object_interpretability_summary.tsv` contains the per-object integrated flags.",
        "- Figures 1–4 summarize scale, stain/encoding, counterfactual, and saliency results.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--expected-objects",
        type=int,
        default=None,
        help="Optional row-count gate; inferred from padding rows when omitted",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    args.output.mkdir(parents=True)

    crop = pd.read_csv(args.input / "crop_reconstruction_validation.tsv", sep="\t")
    padding = pd.read_csv(args.input / "padding_curves.tsv", sep="\t")
    stain = pd.read_csv(args.input / "stain_encoding_sensitivity.tsv", sep="\t")
    counter = pd.read_csv(args.input / "counterfactual_sensitivity.tsv", sep="\t")
    gradcam = pd.read_csv(args.input / "gradcam_metrics.tsv", sep="\t")
    occlusion = pd.read_csv(args.input / "occlusion_metrics.tsv", sep="\t")
    inferred_objects = padding[["sample_id", "object_id"]].drop_duplicates().shape[0]
    expected_objects = args.expected_objects or inferred_objects
    if expected_objects <= 0:
        raise SystemExit("No audit objects found")
    if len(crop) != expected_objects * 2 or not bool(bool_series(crop["exact_match"]).all()):
        raise SystemExit("Crop reconstruction gate failed")
    expected = {
        "padding": expected_objects * 6,
        "stain": expected_objects * 13,
        "counter": expected_objects * 2 * 3,
        "gradcam": expected_objects * 2,
        "occlusion": expected_objects,
    }
    observed = {
        "padding": len(padding),
        "stain": len(stain),
        "counter": len(counter),
        "gradcam": len(gradcam),
        "occlusion": len(occlusion),
    }
    if observed != expected:
        raise SystemExit(f"Row-count gate failed: observed={observed}, expected={expected}")

    padding_objects, by_padding = padding_summary(padding)
    stain_summary_df = stain_summary(stain)
    counter_summary_df, counter_pivot = counterfactual_summary(counter)
    objects = integrated_objects(padding_objects, stain, counter_pivot, gradcam, occlusion)
    padding_objects.to_csv(args.output / "padding_object_summary.tsv", sep="\t", index=False)
    by_padding.to_csv(args.output / "padding_level_summary.tsv", sep="\t", index=False)
    stain_summary_df.to_csv(args.output / "stain_encoding_summary.tsv", sep="\t", index=False)
    counter_summary_df.to_csv(args.output / "counterfactual_summary.tsv", sep="\t", index=False)
    counter_pivot.to_csv(args.output / "counterfactual_object_padding.tsv", sep="\t", index=False)
    objects.to_csv(args.output / "object_interpretability_summary.tsv", sep="\t", index=False)

    plot_padding(padding, by_padding, args.output)
    plot_stain(stain_summary_df, args.output)
    plot_counterfactual(counter, counter_summary_df, counter_pivot, args.output)
    plot_saliency(gradcam, occlusion, objects, args.output)
    build_priority_review_panels(args.input, objects, args.output)

    assessment = markdown_assessment(
        args.input,
        padding_objects,
        stain_summary_df,
        counter_summary_df,
        counter_pivot,
        gradcam,
        occlusion,
        objects,
    )
    (args.output / "INTERPRETABILITY_FINAL_ASSESSMENT.md").write_text(assessment)
    summary = {
        "status": "complete",
        "n_objects": expected_objects,
        "n_slides": int(objects.sample_id.nunique()),
        "crop_hash_matches": expected_objects * 2,
        "padding_stable_objects": int(padding_objects.padding_stable_all.sum()),
        "padding_unstable_objects": int((~padding_objects.padding_stable_all).sum()),
        "review_priority_counts": objects.engineering_review_priority.value_counts().to_dict(),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "SHA256SUMS").open("w") as handle:
        for path in sorted(args.output.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                handle.write(f"{sha256_file(path)}  {path.relative_to(args.output)}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
