"""Research utilities for auditing a multi-model TLS pathology workflow."""

from .audit_metrics import (
    cascade_scores,
    normalized_entropy,
    padding_is_stable,
    saliency_enrichment,
    top1_margin,
)

__all__ = [
    "cascade_scores",
    "normalized_entropy",
    "padding_is_stable",
    "saliency_enrichment",
    "top1_margin",
]

__version__ = "0.1.0"
