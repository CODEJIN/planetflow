"""Unit tests for pipeline/modules/position_refinement.py.

Covers the confidence gate (refine_position_via_cv must return None, forcing
ephemeris-only fallback, whenever the CV re-detection is not both stable
across box sizes AND significant enough), the max_offset_px sanity bound (a
"confident" but implausibly large correction must still be rejected -- see
the module docstring for why confidence alone is not sufficient, per
docs/investigation_shadow_rotational_bias.md section 12-11's finding that
the same re-detection method has its own reproducible bias from real
belt/zone brightness gradients), and both obj_type branches (satellite:
bright signal; shadow: dark signal).

Run directly: python3 tests/test_position_refinement.py
Or via pytest: pytest tests/test_position_refinement.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.modules.position_refinement import (
    RefinementResult,
    apply_offset,
    refine_position_via_cv,
)
from pipeline.modules.satellite_tracker import SatellitePos


def _make_signal(size, true_x, true_y, obj_sigma=3.0, obj_depth=0.30, noise_sigma=0.01, seed=0):
    """Flat background + a single symmetric Gaussian bump/dip + noise.
    Positive obj_depth = bright bump (satellite); negative = dark dip (shadow)."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:size, 0:size]
    img = np.full((size, size), 0.5, dtype=np.float64)
    img += obj_depth * np.exp(-((xs - true_x) ** 2 + (ys - true_y) ** 2) / (2 * obj_sigma ** 2))
    img += rng.normal(0, noise_sigma, size=img.shape)
    return img


def test_confident_satellite_detection_recovers_known_offset():
    """A clear, symmetric bright bump offset from the predicted position must
    be recovered with high/medium confidence and the correct dx/dy sign."""
    size = 80
    x_pred, y_pred = 40.0, 40.0
    true_x, true_y = 43.0, 38.0   # +3px in x, -2px in y from the prediction
    img = _make_signal(size, true_x, true_y, obj_depth=0.30, seed=1)

    result = refine_position_via_cv(img, x_pred, y_pred, r_px=6.0, obj_type="satellite")

    assert result is not None, "clear signal should not fall back to ephemeris-only"
    assert result.confidence in ("high", "medium")
    assert math.isclose(result.dx_px, true_x - x_pred, abs_tol=0.7)
    assert math.isclose(result.dy_px, true_y - y_pred, abs_tol=0.7)


def test_confident_shadow_detection_recovers_known_offset():
    """Same as above but for a dark dip (shadow) -- the obj_type="shadow"
    branch must look for a NEGATIVE (darker-than-background) signal, not
    reuse the satellite (brighter) direction."""
    size = 80
    x_pred, y_pred = 40.0, 40.0
    true_x, true_y = 42.0, 41.0
    img = _make_signal(size, true_x, true_y, obj_depth=-0.30, seed=4)

    result = refine_position_via_cv(img, x_pred, y_pred, r_px=6.0, obj_type="shadow")

    assert result is not None, "clear shadow signal should not fall back to ephemeris-only"
    assert result.confidence in ("high", "medium")
    assert math.isclose(result.dx_px, true_x - x_pred, abs_tol=0.7)
    assert math.isclose(result.dy_px, true_y - y_pred, abs_tol=0.7)


def test_shadow_obj_type_does_not_detect_a_bright_bump():
    """A bright bump must NOT be picked up by the shadow (dark-signal) branch
    -- confirms the two obj_type branches are actually distinct, not just a
    label with no effect on detection direction."""
    size = 80
    img = _make_signal(size, 43.0, 38.0, obj_depth=0.30, seed=1)

    result = refine_position_via_cv(img, 40.0, 40.0, r_px=6.0, obj_type="shadow")

    assert result is None


def test_no_signal_returns_none():
    """Flat background + noise only, no object: must fall back to ephemeris
    (None), not fabricate a random low-confidence offset."""
    size = 80
    rng = np.random.default_rng(2)
    img = np.full((size, size), 0.5, dtype=np.float64) + rng.normal(0, 0.01, size=(size, size))

    result = refine_position_via_cv(img, 40.0, 40.0, r_px=6.0, obj_type="satellite")

    assert result is None


def test_implausibly_large_offset_is_rejected_despite_confidence():
    """A strong, stable signal that sits far from the predicted position
    must still be rejected once it exceeds max_offset_px -- confidence alone
    is not proof of correctness (see module docstring)."""
    size = 220
    x_pred, y_pred = 110.0, 110.0
    r_px = 16.0
    true_x, true_y = 132.0, 110.0   # 22px away -- inside the 1.5*r_px detection radius (24px)
    img = _make_signal(size, true_x, true_y, obj_depth=0.30, seed=3)

    # Sanity: without the cap, this would be confidently detected.
    uncapped = refine_position_via_cv(img, x_pred, y_pred, r_px=r_px, obj_type="satellite", max_offset_px=1000.0)
    assert uncapped is not None and uncapped.confidence in ("high", "medium")

    capped = refine_position_via_cv(img, x_pred, y_pred, r_px=r_px, obj_type="satellite", max_offset_px=20.0)
    assert capped is None


def test_apply_offset_shifts_position_and_recomputes_on_disk():
    disk_cx, disk_cy, disk_r = 100.0, 100.0, 50.0
    # Just inside the disk edge; a +6px x-shift should push it off-disk.
    pos = SatellitePos(name="Io", x_px=145.0, y_px=100.0, on_disk=True, dist_px=45.0)

    shifted = apply_offset([pos], dx=6.0, dy=0.0, disk_cx=disk_cx, disk_cy=disk_cy, disk_r=disk_r)

    assert len(shifted) == 1
    new_pos = shifted[0]
    assert new_pos.x_px == 151.0 and new_pos.y_px == 100.0
    assert math.isclose(new_pos.dist_px, 51.0, abs_tol=1e-9)
    assert new_pos.on_disk is False


def test_apply_offset_preserves_none_entries():
    shifted = apply_offset([None], dx=5.0, dy=5.0, disk_cx=0.0, disk_cy=0.0, disk_r=10.0)
    assert shifted == [None]


if __name__ == "__main__":
    test_confident_satellite_detection_recovers_known_offset()
    print("confident satellite detection recovers known offset: OK")
    test_confident_shadow_detection_recovers_known_offset()
    print("confident shadow detection recovers known offset: OK")
    test_shadow_obj_type_does_not_detect_a_bright_bump()
    print("shadow obj_type does not detect a bright bump: OK")
    test_no_signal_returns_none()
    print("no signal returns None: OK")
    test_implausibly_large_offset_is_rejected_despite_confidence()
    print("implausibly large offset rejected despite confidence: OK")
    test_apply_offset_shifts_position_and_recomputes_on_disk()
    print("apply_offset shifts position and recomputes on_disk: OK")
    test_apply_offset_preserves_none_entries()
    print("apply_offset preserves None entries: OK")
    print("\nAll checks passed.")
