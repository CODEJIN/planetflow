"""Regression test for find_disk_center()'s polarity-aware ring/disk isolation.

Covers the CH4 root-cause bug: a ring band whose brightness is inverted
relative to the disk (or that crosses directly in front of the disk, as
confirmed on real Saturn CH4 data) used to break find_disk_center() outright
because ring-core isolation always assumed the disk was the *brighter* part
of the blob. This test builds synthetic images that reproduce the real
failure geometry (a band crossing the disk equator, extending as ansae
beyond it) in both brightness polarities, and checks real CH4/sibling/Jupiter
data from this session as an end-to-end sanity check.

Run directly: python3 tests/test_disk_polarity.py
Or via pytest: pytest tests/test_disk_polarity.py -v
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.modules import image_io
from pipeline.modules.derotation import (
    FilterPose,
    _find_disk_center_impl,
    find_disk_center,
    resolve_filter_pose,
    resolve_shared_shape,
)

TRUE_SEMI_A = 60.0
TRUE_SEMI_B = 54.0
TRUE_ASPECT = TRUE_SEMI_B / TRUE_SEMI_A


def _make_ring_crossing_synthetic(
    disk_val: float,
    band_val: float,
    size: int = 300,
    cx: float = 150.0,
    cy: float = 150.0,
    semi_a: float = TRUE_SEMI_A,
    semi_b: float = TRUE_SEMI_B,
    band_halfwidth: float = 10.0,
    ansae_len: float = 50.0,
) -> np.ndarray:
    """Synthetic oblate disk with a ring band crossing its equator and
    extending as ansae beyond the limb — reproduces the real Saturn CH4
    geometry (ring literally overlaying part of the disk silhouette,
    confirmed visually on real session data), not just a separate annulus
    around it.
    """
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    rd = np.sqrt(((xx - cx) / semi_a) ** 2 + ((yy - cy) / semi_b) ** 2)
    disk = (rd <= 1.0).astype(np.float32) * disk_val
    band = (
        (np.abs(yy - cy) <= band_halfwidth) & (np.abs(xx - cx) <= semi_a + ansae_len)
    ).astype(np.float32) * band_val
    img = np.clip(np.maximum(disk, band), 0.0, 1.0)
    return cv2.GaussianBlur(img, (5, 5), 1.5)


def test_normal_polarity_disk_brighter_than_ring():
    """Disk brighter than the crossing ring band (the common case) — must
    still isolate the disk correctly (no regression from the polarity fix)."""
    img = _make_ring_crossing_synthetic(disk_val=0.6, band_val=0.3)
    cx, cy, semi_a, semi_b, angle, confidence, shape_reliable = _find_disk_center_impl(img)
    assert shape_reliable, "bright-core isolation should succeed when disk is brighter"
    assert abs(semi_a - TRUE_SEMI_A) < 8.0, f"semi_a={semi_a} too far from true {TRUE_SEMI_A}"
    assert abs(semi_b / semi_a - TRUE_ASPECT) < 0.1


def test_inverted_polarity_disk_darker_than_ring():
    """Disk darker than the crossing ring band (Saturn CH4 band) — this is
    the exact bug: bright-core isolation must fail gracefully and the
    radial-limb fallback must still recover a plausible disk radius."""
    img = _make_ring_crossing_synthetic(disk_val=0.3, band_val=0.6)
    cx, cy, semi_a, semi_b, angle, confidence, shape_reliable = _find_disk_center_impl(img)
    assert not shape_reliable, "oblateness is not independently known via the fallback path"
    # 0.5 = gradient search found >= 8 valid edge crossings (this clean
    # synthetic case); 0.3 = it didn't and the geometric seed was used
    # unconfirmed (the common case on real, noisier Saturn CH4 data — see
    # test_ch4_real_data_matches_sibling_scale). Both are valid non-failure
    # outcomes of this fallback; confidence==0.0 (total failure) is not.
    assert confidence in (0.3, 0.5), f"expected a fallback confidence tier, got {confidence}"
    assert abs(semi_a - TRUE_SEMI_A) < 10.0, f"semi_a={semi_a} too far from true {TRUE_SEMI_A}"


def test_inverted_polarity_would_have_failed_the_old_way():
    """Sanity check that this synthetic case genuinely exercises the bug: the
    raw bright-core-only isolation (no fallback) must produce an implausible
    result, or this test wouldn't be testing anything real."""
    img = _make_ring_crossing_synthetic(disk_val=0.3, band_val=0.6)
    arr8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    thresh_val, _ = cv2.threshold(arr8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary = cv2.threshold(arr8, max(1, int(thresh_val * 0.9)), 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)
    roi_vals = arr8[y : y + bh, x : x + bw]
    roi_mask = binary[y : y + bh, x : x + bw] > 0
    vals = roi_vals[roi_mask]
    core_thv = np.percentile(vals, 60.0)
    core_bin = (((roi_vals >= core_thv) & roi_mask).astype(np.uint8)) * 255
    core_contours, _ = cv2.findContours(core_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    core_largest = max(core_contours, key=cv2.contourArea)
    (_, _), (cma, cmi), _ = cv2.fitEllipse(core_largest + np.array([x, y], dtype=core_largest.dtype))
    old_semi_a, old_semi_b = (cma / 2, cmi / 2) if cma >= cmi else (cmi / 2, cma / 2)
    old_aspect = old_semi_b / old_semi_a if old_semi_a > 0 else 0.0
    assert old_aspect < 0.4 or abs(old_semi_a - TRUE_SEMI_A) > 15.0, (
        "synthetic case doesn't reproduce the bug — bright-core-only isolation "
        f"gave semi_a={old_semi_a:.1f} aspect={old_aspect:.3f}, which looks fine"
    )


def test_ch4_real_data_matches_sibling_scale():
    """Real CH4 frames from this session must now report a stable, plausible
    radius (previously broken at semi_a~145-148px, aspect~0.16-0.34)."""
    files = sorted(glob.glob("Saturn_Data/step02_lucky_stack/*-CH4-*.tif"))
    if not files:
        return  # session data not present in this environment — skip
    semi_as = []
    for p in files:
        raw = image_io.read_tif(p)
        lum = (raw if raw.ndim == 2 else raw.mean(axis=2)).astype(np.float32)
        cx, cy, semi_a, semi_b, angle, confidence, shape_reliable = _find_disk_center_impl(lum)
        assert confidence > 0.0, f"{p}: total detection failure"
        assert not shape_reliable, f"{p}: expected the radial-limb fallback for this session's CH4 data"
        semi_as.append(semi_a)
    # Consistent across the whole session (was wildly ring-sized before: ~145-148px)
    assert max(semi_as) - min(semi_as) < 5.0, f"CH4 semi_a not stable across session: {semi_as}"
    assert 40.0 < np.median(semi_as) < 65.0, f"CH4 semi_a implausible: median={np.median(semi_as)}"


def test_saturn_sibling_filters_unaffected():
    """IR/R/G/B (non-CH4) Saturn filters must keep their own reliable shape,
    with plausible Saturn-disk oblateness."""
    found_any = False
    for filt in ["IR", "R", "G", "B"]:
        for p in sorted(glob.glob(f"Saturn_Data/step02_lucky_stack/*-{filt}-*.tif"))[:3]:
            found_any = True
            raw = image_io.read_tif(p)
            lum = (raw if raw.ndim == 2 else raw.mean(axis=2)).astype(np.float32)
            cx, cy, semi_a, semi_b, angle, confidence, shape_reliable = _find_disk_center_impl(lum)
            assert shape_reliable, f"{p}: expected reliable bright-core isolation"
            aspect = semi_b / semi_a
            assert 0.75 < aspect < 1.0, f"{p}: implausible aspect {aspect}"
    if not found_any:
        return  # session data not present in this environment — skip


def test_jupiter_byte_identical_to_ringless_path():
    """Jupiter (no ring, raw_aspect >= 0.80) must never enter the new
    polarity/fallback branch at all — confidence must be exactly 1.0."""
    files = sorted(glob.glob("/data/astro_test_bak/Mono_Sample/*.tif"))
    if not files:
        return  # sample data not present in this environment — skip
    import tifffile

    for p in files[:8]:
        img = tifffile.imread(p)
        lum = (img if img.ndim == 2 else img.mean(axis=2)).astype(np.float32)
        if lum.max() > 1.5:
            lum = lum / lum.max()
        cx, cy, semi_a, semi_b, angle, confidence, shape_reliable = _find_disk_center_impl(lum)
        assert confidence == 1.0, f"{p}: Jupiter should never trigger ring-core isolation"
        assert shape_reliable


def test_resolve_shared_shape_and_pose_end_to_end():
    """resolve_shared_shape()/resolve_filter_pose() combine correctly: CH4
    borrows aspect_ratio/angle but keeps its own (larger-confidence-checked)
    pose and its own semi_major."""
    normal = _make_ring_crossing_synthetic(disk_val=0.6, band_val=0.3)
    inverted = _make_ring_crossing_synthetic(disk_val=0.3, band_val=0.6, cx=140.0, cy=160.0)
    fits = {
        "R": _find_disk_center_impl(normal),
        "CH4": _find_disk_center_impl(inverted),
    }
    result = resolve_shared_shape(fits)
    assert result is not None
    shape, source = result
    assert source == "R"

    probe_fit = fits["R"]
    probe_pose = FilterPose(probe_fit[0], probe_fit[1])
    pose, method = resolve_filter_pose(
        fits["CH4"], lum=inverted, probe_lum=normal, probe_pose=probe_pose, probe_semi_major_px=probe_fit[2]
    )
    assert method == "own_detection", "CH4's own pose is trustworthy here (confidence=0.5, not 0)"
    # CH4's own detected center (140, 160), NOT borrowed from R's (150, 150).
    assert abs(pose.center_x_px - 140.0) < 5.0
    assert abs(pose.center_y_px - 160.0) < 5.0


def test_resolve_filter_pose_registration_sign():
    """registered_to_probe (total detection failure, confidence==0.0) must
    recover this filter's TRUE own center, not the probe's center reflected
    through itself — regression test for a confirmed sign-inversion bug."""
    size = 300
    probe_cx, probe_cy = 150.0, 150.0
    true_own_cx, true_own_cy = 155.0, 147.0  # a plausible few-px filter-wheel offset

    def _single_disk(cx, cy, r=60.0):
        yy, xx = np.mgrid[:size, :size].astype(np.float32)
        img = (np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) <= r).astype(np.float32) * 0.6
        return cv2.GaussianBlur(img, (5, 5), 1.5)

    probe_lum = _single_disk(probe_cx, probe_cy)
    own_lum = _single_disk(true_own_cx, true_own_cy)

    # A fit with confidence == 0.0 forces the registration fallback,
    # regardless of the (irrelevant, known-bad) cx/cy/semi_a it carries.
    failed_fit = (0.0, 0.0, 60.0, 60.0, 0.0, 0.0, False)
    probe_pose = FilterPose(probe_cx, probe_cy)

    pose, method = resolve_filter_pose(
        failed_fit, lum=own_lum, probe_lum=probe_lum, probe_pose=probe_pose, probe_semi_major_px=60.0
    )
    assert method == "registered_to_probe"
    assert abs(pose.center_x_px - true_own_cx) < 1.0, (
        f"got {pose.center_x_px:.2f}, expected ~{true_own_cx} "
        f"(sign-flipped bug would give ~{2*probe_cx - true_own_cx:.2f})"
    )
    assert abs(pose.center_y_px - true_own_cy) < 1.0
    assert pose.semi_major_px == 60.0


def test_public_api_unchanged():
    """find_disk_center()'s public 5-tuple contract must be untouched."""
    img = _make_ring_crossing_synthetic(disk_val=0.6, band_val=0.3)
    result = find_disk_center(img)
    assert len(result) == 5
    assert all(isinstance(v, float) for v in result)


def test_one_sided_edge_rays_are_rejected():
    """_gradient_disk_r(min_quadrants=3) must not accept an "edge" seen only
    on one side of the centre -- the 271026 CH4 failure, where every valid
    ray (25°-150°) caught the lower edge of the ring band crossing the globe
    and the fallback reported r≈30.7 px for a ≈63 px disk."""
    from pipeline.modules.derotation import _gradient_disk_r

    size, c = 300, 150.0
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    # Bright band below centre ending at y = c + 30; above it a smooth
    # ramp with no edge anywhere in the search window.
    img = np.where((yy > c) & (yy < c + 30), 0.8, 0.0) + 0.3 * np.clip(1 - np.hypot(xx - c, yy - c) / 140, 0, 1)
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 2.0)
    r, n = _gradient_disk_r(img, c, c, 50.0, search_frac=(0.5, 1.6), return_n_valid=True)
    r_q, n_q = _gradient_disk_r(img, c, c, 50.0, search_frac=(0.5, 1.6), return_n_valid=True,
                                min_quadrants=3)
    assert n >= 8 and r < 40.0, f"setup should reproduce the one-sided edge (got r={r:.1f}, n={n})"
    assert n_q == 0 and r_q == 50.0, f"one-sided edge must be rejected, got r={r_q:.1f}, n={n_q}"


def test_real_saturn_271026_ch4_radius():
    """Real CH4 frames of D:/Astro/Sat/271026 (disk darker than the rings,
    ring band crossing the globe): r≈63 px, not the r≈30.7 px the zero-
    padded limb search used to report in every frame."""
    files = sorted(glob.glob("D:/Astro/Sat/271026/step02_lucky_stack/*-CH4-*.tif"))
    if not files:
        return
    radii = [_find_disk_center_impl(image_io.read_tif(p))[2] for p in files[::3]]
    assert 58.0 < float(np.median(radii)) < 68.0, f"CH4 radius implausible: {radii}"
    assert max(radii) - min(radii) < 6.0, f"CH4 radius unstable: {radii}"


if __name__ == "__main__":
    tests = [
        test_normal_polarity_disk_brighter_than_ring,
        test_inverted_polarity_disk_darker_than_ring,
        test_inverted_polarity_would_have_failed_the_old_way,
        test_ch4_real_data_matches_sibling_scale,
        test_saturn_sibling_filters_unaffected,
        test_jupiter_byte_identical_to_ringless_path,
        test_resolve_shared_shape_and_pose_end_to_end,
        test_resolve_filter_pose_registration_sign,
        test_public_api_unchanged,
        test_one_sided_edge_rays_are_rejected,
        test_real_saturn_271026_ch4_radius,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print("\nAll checks passed.")
