#!/usr/bin/env python3
"""Synthetic acceptance tests for boundary graphs, alignment and controls."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tls_audit.spatial_features import (
    CLASSES, boundary_pair_features, build_label_grid, lym_bridge_count,
    polygon_center_um, proximity_pair_features, select_controls, window_features,
)


def grid(rows):
    return np.asarray([[CLASSES.index(x) if x != "." else -1 for x in row] for row in rows], dtype=np.int16)


def run():
    # 1: Empty/unassigned field is safe and produces zeros.
    g = grid([[".", "."], [".", "."]])
    f = boundary_pair_features(g, "LYM", "TUM", 256)
    assert f["lym_tum_contact_pair_count"] == 0 and f["lym_tum_interface_um"] == 0

    # 2: Isolated tissues do not create boundary edges.
    g = grid([["LYM", ".", "TUM"]])
    assert boundary_pair_features(g, "LYM", "TUM", 256)["lym_tum_contact_pair_count"] == 0

    # 3: Direct side adjacency creates one pair and one tile-edge interface.
    g = grid([["LYM", "TUM"]])
    f = boundary_pair_features(g, "LYM", "TUM", 256)
    assert f["lym_tum_contact_pair_count"] == 1 and f["lym_tum_interface_um"] == 256

    # 4: One LYM component bridging tumour and stroma is counted once.
    g = grid([["TUM", "LYM", "STR"], ["TUM", "LYM", "STR"]])
    assert lym_bridge_count(g) == 1

    # 5: Sparse coordinate gaps remain gaps; no accidental compressed adjacency.
    t = pd.DataFrame({
        "center_x_um": [128, 5 * 256 + 128],
        "center_y_um": [128, 128],
        "predicted_class": ["LYM", "TUM"],
        "confidence": [1.0, 1.0],
    })
    sparse, _ = build_label_grid(t, 0, 256)
    assert sparse.shape == (1, 6)
    assert boundary_pair_features(sparse, "LYM", "TUM", 256)["lym_tum_contact_pair_count"] == 0

    # 6: Physical centre alignment selects the intended tile, not its transpose.
    t = pd.DataFrame({
        "center_x_um": [3 * 256 + 128],
        "center_y_um": [1 * 256 + 128],
        "predicted_class": ["LYM"],
        "confidence": [1.0],
    })
    aligned, _ = build_label_grid(t, 0, 256)
    assert aligned[1, 3] == CLASSES.index("LYM")
    wf = window_features(aligned, 3 * 256 + 128, 1 * 256 + 128, 200, 256)
    assert wf["fraction_LYM"] == 1.0

    # 7: One intervening cell is near-contact at 512 um but not direct contact.
    g = grid([["LYM", ".", "TUM"]])
    assert boundary_pair_features(g, "LYM", "TUM", 256)["lym_tum_contact_pair_count"] == 0
    assert proximity_pair_features(g, "LYM", "TUM", 2, 256)["lym_tum_proximity_pair_count_512um"] == 1

    # 8: Confidence threshold turns a low-confidence tile into unassigned.
    t = pd.DataFrame({
        "center_x_um": [128], "center_y_um": [128],
        "predicted_class": ["LYM"], "confidence": [0.49],
    })
    low, _ = build_label_grid(t, .5, 256)
    assert low[0, 0] == -1

    # 9: Three controls are selected for one TLS and are unique when feasible.
    g = np.full((40, 40), CLASSES.index("NORM"), dtype=np.int16)
    tls = pd.DataFrame({"object_id": ["x"], "center_x_um": [20 * 112], "center_y_um": [20 * 112]})
    controls = select_controls(g, tls, 112, radius_um=500, exclusion_um=750, controls_per_tls=3)
    assert len(controls) == 3
    assert len({(x["center_x_um"], x["center_y_um"]) for x in controls}) == 3
    assert [x["control_id"] for x in controls] == [1, 2, 3]

    # 10: Area centroid is not biased by uneven sampling of a polygon edge.
    uneven_rectangle = np.asarray([
        [0, 0], [10, 0], [10, 2], [10, 10], [0, 10],
    ], dtype=float)
    area_center = polygon_center_um(uneven_rectangle, mpp=0.5)
    vertex_center = polygon_center_um(uneven_rectangle, mpp=0.5, method="vertex_mean")
    assert np.allclose(area_center, (2.5, 2.5))
    assert not np.allclose(area_center, vertex_center)

    # 11: Degenerate polygons use a deterministic vertex-mean fallback.
    line = np.asarray([[0, 0], [2, 0], [4, 0]], dtype=float)
    assert np.allclose(polygon_center_um(line, mpp=0.5), (1.0, 0.0))
    print("PASS: 11 synthetic graph/alignment/control/centroid tests")


if __name__ == "__main__":
    run()
