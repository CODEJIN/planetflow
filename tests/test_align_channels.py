"""Regression tests for composite channel alignment (estimate_channel_shift /
align_channels / align_lrgb).

Background: alignment used to run cv2.phaseCorrelate() on raw intensity,
gated by a disk-region NCC. On real data that lost the peak entirely on
some pairs and locked onto plausible-but-wrong offsets on others (Jupiter
B in ~25% of windows; Saturn 271026 G/B up to 5 px off, which the gate then
either accepted or "fixed" by applying no shift at all, leaving the
tracking drift between filter captures -- up to ~11 px -- uncorrected).
The visible result was green/magenta double rings on Saturn.

Alignment is now whole-frame gradient-magnitude NCC (no disk-ROI crop),
measured on the unsharpened step04 stacks, and LRGB registers the colour
channels to each other before moving them to L as one group.

Run directly: python tests/test_align_channels.py
Or via pytest: pytest tests/test_align_channels.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.modules import image_io
from pipeline.modules.composite import (
    align_channels,
    align_lrgb,
    apply_shift,
    estimate_channel_shift,
)


def _make_disk(h=200, w=200, cx=100.0, cy=100.0, r=70.0, seed=0, noise=0.02):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    mask = np.hypot(xx - cx, yy - cy) < r
    belts = 0.5 + 0.3 * np.sin(yy / 9.0) + 0.1 * np.sin(xx / 13.0)
    img = np.where(mask, belts, 0.0) + noise * rng.standard_normal((h, w))
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _displace(img, dx, dy):
    """Content moved by (dx, dy); the correcting shift is (-dx, -dy)."""
    return apply_shift(img, dx, dy)


def test_known_shift_is_recovered():
    ref = _make_disk(seed=1)
    ch = _displace(_make_disk(seed=2), 1.3, -0.7)
    _, shifts = align_channels({"ref": ref, "ch": ch}, "ref", max_shift_px=10.0)
    dx, dy = shifts["ch"]
    assert abs(dx + 1.3) < 0.1 and abs(dy - 0.7) < 0.1, f"expected ≈(-1.3, +0.7), got {shifts['ch']}"


def test_large_drift_is_recovered():
    """Tracking drift between filter captures reached ~11 px on real data."""
    ref = _make_disk(seed=1)
    ch = _displace(_make_disk(seed=2), -4.2, 11.5)
    _, shifts = align_channels({"ref": ref, "ch": ch}, "ref", max_shift_px=15.0)
    dx, dy = shifts["ch"]
    assert abs(dx - 4.2) < 0.1 and abs(dy + 11.5) < 0.1, shifts["ch"]


def test_different_brightness_profile_still_aligns():
    """Filters differ in limb darkening and feature contrast (IR vs. B);
    edges must still be matched, not brightness."""
    ref = _make_disk(seed=1)
    yy, xx = np.mgrid[0:200, 0:200]
    limb = np.clip(1.0 - (np.hypot(xx - 100, yy - 100) / 70.0) ** 2, 0, 1) ** 0.5
    other = (_make_disk(seed=3) ** 2 * (0.3 + 0.7 * limb)).astype(np.float32)
    ch = _displace(other, 2.5, 1.5)
    est = estimate_channel_shift(ref, ch)
    assert est is not None
    assert abs(est[0] + 2.5) < 0.3 and abs(est[1] + 1.5) < 0.3, est


def test_sky_plateau_edge_does_not_bias():
    """step04 stacks carry a sky plateau (≈0.03) over each filter's own
    coverage rectangle; its hard edge must not pull the estimate."""
    ref = _make_disk(seed=1, noise=0.005)
    ch = _displace(_make_disk(seed=2, noise=0.005), 3.0, -2.0)
    ref[20:185, 10:190] += 0.03
    ch[5:170, 30:195] += 0.03
    est = estimate_channel_shift(ref, ch)
    assert est is not None
    assert abs(est[0] + 3.0) < 0.2 and abs(est[1] - 2.0) < 0.2, est


def test_offset_beyond_search_is_rejected():
    ref = _make_disk(seed=1)
    ch = _displace(_make_disk(seed=2), 9.0, 0.0)
    _, shifts = align_channels({"ref": ref, "ch": ch}, "ref", max_shift_px=5.0)
    assert shifts["ch"] == (0.0, 0.0), shifts["ch"]


def test_no_common_structure_is_rejected():
    rng = np.random.default_rng(0)
    ref = _make_disk(seed=1)
    noise = rng.random((200, 200)).astype(np.float32)
    assert estimate_channel_shift(ref, noise) is None


def test_reference_channel_always_zero_shift():
    ref = _make_disk(seed=3)
    other = _make_disk(seed=4)
    _, shifts = align_channels({"ref": ref, "other": other}, "ref", max_shift_px=10.0)
    assert shifts["ref"] == (0.0, 0.0)


def test_estimate_from_is_measured_but_channels_are_shifted():
    """Shifts measured on estimate_from (step04) must be applied to the
    channels themselves (step05), which are left otherwise untouched."""
    ref4 = _make_disk(seed=1)
    ch4 = _displace(_make_disk(seed=2), 2.0, 1.0)
    ref5 = np.full_like(ref4, 0.5)          # uninformative on purpose
    ch5 = _displace(_make_disk(seed=5), 2.0, 1.0)
    aligned, shifts = align_channels({"ref": ref5, "ch": ch5}, "ref",
                                     estimate_from={"ref": ref4, "ch": ch4})
    assert abs(shifts["ch"][0] + 2.0) < 0.1 and abs(shifts["ch"][1] + 1.0) < 0.1, shifts
    assert aligned["ref"] is ref5
    np.testing.assert_allclose(aligned["ch"], apply_shift(ch5, *shifts["ch"]), atol=1e-6)


def test_lrgb_registers_colours_to_each_other():
    """Colour channels must end up registered to each other, and the group
    to L, with each channel's shift = its own + the group's."""
    L = _make_disk(seed=10) ** 0.5           # different contrast from colours
    R = _displace(_make_disk(seed=11), 1.0, 2.0)
    G = _displace(_make_disk(seed=12), -3.0, 4.0)
    B = _displace(_make_disk(seed=13), 2.0, -1.0)
    chans = {"IR": L, "R": R, "G": G, "B": B}
    _, shifts = align_lrgb(chans, "IR", ["R", "G", "B"], "R")
    assert shifts["IR"] == (0.0, 0.0)
    for k, (tx, ty) in {"R": (1.0, 2.0), "G": (-3.0, 4.0), "B": (2.0, -1.0)}.items():
        assert abs(shifts[k][0] + tx) < 0.15 and abs(shifts[k][1] + ty) < 0.15, (k, shifts[k])


# ── Real-data regression (skip silently if absent) ──────────────────────────

def test_real_saturn_271026_window06():
    """Visually confirmed fringe-free shifts for D:/Astro/Sat/271026 w06
    (the old phase correlation measured G at (-3.4, -2.2), 6 px off)."""
    d = Path("D:/Astro/Sat/271026/step04_derotated/window_06")
    if not d.is_dir():
        return
    m = {f: image_io.read_tif(d / f"{f}_derotated.tif") for f in ("R", "G", "B")}
    _, shifts = align_channels(m, "R", max_shift_px=15.0)
    for k, (ex, ey) in {"G": (-3.2, -7.9), "B": (-2.4, -5.8)}.items():
        dx, dy = shifts[k]
        assert abs(dx - ex) < 0.5 and abs(dy - ey) < 0.5, (k, shifts[k])


if __name__ == "__main__":
    tests = [
        test_known_shift_is_recovered,
        test_large_drift_is_recovered,
        test_different_brightness_profile_still_aligns,
        test_sky_plateau_edge_does_not_bias,
        test_offset_beyond_search_is_rejected,
        test_no_common_structure_is_rejected,
        test_reference_channel_always_zero_shift,
        test_estimate_from_is_measured_but_channels_are_shifted,
        test_lrgb_registers_colours_to_each_other,
        test_real_saturn_271026_window06,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print("\nAll checks passed.")
