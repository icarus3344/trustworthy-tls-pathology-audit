from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tls_audit.audit_metrics import (
    cascade_scores,
    normalized_entropy,
    padding_is_stable,
    saliency_enrichment,
    top1_margin,
)


class AuditMetricTests(unittest.TestCase):
    def test_cascade_scores_sum_to_one(self) -> None:
        values = cascade_scores(
            {"Immature": 0.2, "Mature": 0.8},
            {"Primary": 0.75, "Secondary": 0.25},
        )
        self.assertAlmostEqual(sum(values.values()), 1.0)
        self.assertAlmostEqual(values["E-TLS"], 0.2)
        self.assertAlmostEqual(values["P-TLS"], 0.6)
        self.assertAlmostEqual(values["S-TLS"], 0.2)

    def test_entropy_and_margin(self) -> None:
        uniform = {"E": 1.0, "P": 1.0, "S": 1.0}
        self.assertAlmostEqual(normalized_entropy(uniform), 1.0)
        self.assertAlmostEqual(top1_margin({"E": 0.1, "P": 0.7, "S": 0.2}), 0.5)

    def test_padding_stability(self) -> None:
        self.assertTrue(padding_is_stable(["P-TLS"] * 6))
        self.assertFalse(padding_is_stable(["P-TLS", "S-TLS"]))

    def test_saliency_enrichment(self) -> None:
        value = saliency_enrichment([3.0, 1.0, 0.0, 0.0], [True, False, False, False])
        self.assertAlmostEqual(value, 3.0)
        self.assertTrue(math.isnan(saliency_enrichment([0.0, 0.0], [True, False])))


if __name__ == "__main__":
    unittest.main()
