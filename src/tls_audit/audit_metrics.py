"""Dependency-free metrics used by the TLS maturity reliability audit."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

FINAL_CLASSES = ("E-TLS", "P-TLS", "S-TLS")


def _normalise(values: Mapping[str, float], names: Sequence[str]) -> dict[str, float]:
    cleaned = {name: max(0.0, float(values[name])) for name in names}
    total = sum(cleaned.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Probabilities must have a finite positive sum")
    return {name: value / total for name, value in cleaned.items()}


def cascade_scores(
    step1: Mapping[str, float],
    step2: Mapping[str, float],
) -> dict[str, float]:
    """Return the explicitly derived E/P/S engineering score.

    Step 1 is expected to contain Immature and Mature. Step 2 is expected to
    contain Primary and Secondary. These values are not calibrated pathology
    probabilities and are not presented as an upstream author-defined rule.
    """

    first = _normalise(step1, ("Immature", "Mature"))
    second = _normalise(step2, ("Primary", "Secondary"))
    result = {
        "E-TLS": first["Immature"],
        "P-TLS": first["Mature"] * second["Primary"],
        "S-TLS": first["Mature"] * second["Secondary"],
    }
    return _normalise(result, FINAL_CLASSES)


def normalized_entropy(probabilities: Mapping[str, float]) -> float:
    """Return Shannon entropy divided by log(number of non-empty classes)."""

    probabilities = _normalise(probabilities, tuple(probabilities))
    if len(probabilities) <= 1:
        return 0.0
    entropy = -sum(value * math.log(value) for value in probabilities.values() if value > 0)
    return entropy / math.log(len(probabilities))


def top1_margin(probabilities: Mapping[str, float]) -> float:
    """Return the gap between the two largest normalized probabilities."""

    ordered = sorted(_normalise(probabilities, tuple(probabilities)).values())
    if len(ordered) < 2:
        raise ValueError("At least two classes are required")
    return ordered[-1] - ordered[-2]


def padding_is_stable(labels: Iterable[str]) -> bool:
    """Return True only when every evaluated crop context has one label."""

    labels = tuple(labels)
    if not labels:
        raise ValueError("At least one label is required")
    return len(set(labels)) == 1


def saliency_enrichment(
    saliency: Sequence[float],
    inside_polygon: Sequence[bool],
) -> float:
    """Compare saliency mass inside a polygon with its area fraction.

    A value above one means that the map is enriched inside the polygon
    relative to a uniform map. This is an evidence-localization metric, not a
    correctness or causal metric.
    """

    if len(saliency) != len(inside_polygon) or not saliency:
        raise ValueError("Saliency and mask must have equal non-zero length")
    weights = [max(0.0, float(value)) for value in saliency]
    total = sum(weights)
    area = sum(bool(value) for value in inside_polygon) / len(inside_polygon)
    if total <= 0 or area <= 0:
        return math.nan
    inside = sum(weight for weight, keep in zip(weights, inside_polygon) if keep) / total
    return inside / area
