"""Regression tests for discrete few-degree camera-rotation detection and
correction (detect_orientation_rotation_map / read_tif_oriented rot_deg).

Context: re-seating the camera or re-acquiring the target mid-session can
leave the image rotated by a few degrees. Confirmed on real data
(D:/Astro/Sat/271026, IR, 31 frames): a +5.1° step across the 13:27→14:04
re-acquisition gap, with no 180° flip. Adjacent no-step pairs measured
< 0.5° typically and at most 1.0° (Jupiter 260407, features moving between
frames), so the 2° step threshold sits clear of that noise. These tests
guard detection of a real step, the no-op for sessions without one, and
that slow continuous drift is never "corrected".

Run directly: python tests/test_orientation_rotation.py
Or via pytest: pytest tests/test_orientation_rotation.py -v
"""
from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.modules import image_io, quality
from pipeline.modules.derotation import (
    _pairwise_rotation_estimate,
    _RotationProbe,
    detect_orientation_rotation_map,
)
from test_meridian_flip import _PatchedReadTif

H, W = 300, 300
CX, CY = 150.0, 150.0


def make_ringed_planet(seed: int = 0, noise_amp: float = 0.02) -> np.ndarray:
    """Saturn-like frame: belted globe plus a thin tilted ring ellipse.
    The ring and fine belts survive the detector's σ=3 high-pass and carry
    the orientation signal (a plain disk's circular limb does not)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = xx - CX, yy - CY
    globe = (dx ** 2 + dy ** 2) < 50 ** 2
    img = np.zeros((H, W), np.float32)
    img[globe] = (0.6 + 0.15 * np.sin(dy / 4.0) + 0.08 * np.sin(dy / 9.0 + 1.0))[globe]
    rr = np.sqrt((dx / 110.0) ** 2 + (dy / 25.0) ** 2)
    ring = (np.abs(rr - 0.9) < 0.06) & ~(globe & (dy < 0))
    img[ring] = 0.8
    rng = np.random.default_rng(seed)
    img = img + noise_amp * rng.standard_normal((H, W)).astype(np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    M = cv2.getRotationMatrix2D((CX, CY), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)


def _build_groups(
    angles: List[float],
    flips: "List[bool] | None" = None,
    filters: List[str] = ["IR", "R"],
    cycle_sec: float = 60.0,
    gap_after: "int | None" = None,
) -> Tuple[Dict[str, List[Tuple[Path, dict]]], Dict[str, np.ndarray], Dict[str, float]]:
    """Synthetic session where frame i (every filter) is the belted disk
    rotated by angles[i] (cv2 convention) and optionally flipped 180°.
    Returns (groups, images, applied_angle_per_stem)."""
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    groups: Dict[str, List[Tuple[Path, dict]]] = {f: [] for f in filters}
    images: Dict[str, np.ndarray] = {}
    applied: Dict[str, float] = {}
    for i, angle in enumerate(angles):
        extra = 1800.0 if gap_after is not None and i > gap_after else 0.0
        for j, filt in enumerate(filters):
            ts = t0 + timedelta(seconds=i * cycle_sec + j * (cycle_sec / len(filters)) + extra)
            stem = f"frame_{filt}_{i:03d}"
            path = Path(f"/synthetic/{stem}.tif")
            img = _rotate(make_ringed_planet(seed=1000 * i + j), angle)
            if flips is not None and flips[i]:
                img = np.ascontiguousarray(img[::-1, ::-1])
            images[str(path)] = img
            applied[stem] = angle
            groups[filt].append((path, {"stem": stem, "timestamp": ts,
                                          "filter": filt, "target": "Jup"}))
    return groups, images, applied


def test_pairwise_estimate_recovers_known_rotation():
    """b = a rotated by +6° → the estimate must be the angle that undoes it
    (-6°), within 0.3°, with a clear NCC gain over no rotation."""
    a = make_ringed_planet(seed=1)
    b = _rotate(make_ringed_planet(seed=2), 6.0)
    angle, ncc_best, ncc_zero = _pairwise_rotation_estimate(_RotationProbe(a), _RotationProbe(b))
    assert abs(angle - (-6.0)) < 0.3, f"expected ≈ -6.0°, got {angle:+.2f}°"
    assert ncc_best - ncc_zero > 0.1, (ncc_best, ncc_zero)


def test_single_step_is_detected_and_corrected():
    """+6° applied from frame 12 of 20: the longer first segment is the
    reference (0°), every later frame of every filter gets ≈ -6°."""
    angles = [0.0] * 12 + [6.0] * 8
    groups, images, applied = _build_groups(angles, gap_after=11)
    with _PatchedReadTif(images):
        rot_map = detect_orientation_rotation_map(groups)
    assert rot_map, "expected a non-empty rotation map"
    for stem, a in applied.items():
        assert abs(rot_map[stem] - (-a)) < 0.4, f"{stem}: expected {-a:+.1f}°, got {rot_map[stem]:+.2f}°"


def test_reference_is_longest_segment():
    """Step at frame 5 of 20: the later, longer segment is the reference,
    so the first 5 frames get the correction (≈ +6°) and the rest 0°."""
    angles = [-6.0] * 5 + [0.0] * 15
    groups, images, applied = _build_groups(angles, gap_after=4)
    with _PatchedReadTif(images):
        rot_map = detect_orientation_rotation_map(groups)
    for stem, a in applied.items():
        assert abs(rot_map[stem] - (-a)) < 0.4, f"{stem}: expected {-a:+.1f}°, got {rot_map[stem]:+.2f}°"


def test_flip_plus_residual_rotation():
    """A 180° flip together with a +5° tilt: with the flip map supplied the
    rotation detector sees only the residual tilt and corrects it."""
    angles = [0.0] * 12 + [5.0] * 8
    flips = [False] * 12 + [True] * 8
    groups, images, applied = _build_groups(angles, flips=flips, gap_after=11)
    flip_map = {m["stem"]: flips[int(m["stem"][-3:])] for f in groups for _, m in groups[f]}
    with _PatchedReadTif(images):
        rot_map = detect_orientation_rotation_map(groups, flip_map)
    for stem, a in applied.items():
        # A 180° rotation commutes with any rotation, so the residual
        # correction after undoing the flip is still -a.
        assert abs(rot_map[stem] - (-a)) < 0.4, f"{stem}: expected {-a:+.1f}°, got {rot_map[stem]:+.2f}°"


def test_no_false_positive_without_step():
    groups, images, _ = _build_groups([0.0] * 20)
    with _PatchedReadTif(images):
        rot_map = detect_orientation_rotation_map(groups)
    assert rot_map == {}, f"expected no rotation map, got {len(rot_map)} entries"


def test_continuous_drift_is_not_corrected():
    """0.3°/frame over 20 frames (5.7° total, like slow field rotation):
    no single adjacent step reaches the threshold, so nothing is corrected."""
    groups, images, _ = _build_groups([0.3 * i for i in range(20)])
    with _PatchedReadTif(images):
        rot_map = detect_orientation_rotation_map(groups)
    assert rot_map == {}, f"continuous drift must not be corrected, got {len(rot_map)} entries"


def test_read_tif_oriented_rotation():
    """rot_deg=0 must be byte-identical to read_tif; a non-zero rot_deg
    must undo a known rotation about the disk centre."""
    img = make_ringed_planet(seed=42)
    with tempfile.TemporaryDirectory() as d:
        p0 = Path(d) / "plain.tif"
        p1 = Path(d) / "rotated.tif"
        image_io.write_tif_16bit(img, p0)
        image_io.write_tif_16bit(np.clip(_rotate(img, 5.0), 0, 1), p1)
        np.testing.assert_array_equal(image_io.read_tif_oriented(p0, False, 0.0), image_io.read_tif(p0))
        restored = image_io.read_tif_oriented(p1, False, -5.0)
        base = image_io.read_tif(p0)
        yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        inner = (xx - CX) ** 2 + (yy - CY) ** 2 < 45 ** 2
        err = np.abs(restored - base)[inner].mean()
        assert err < 0.02, f"rotation not undone (mean abs err {err:.4f})"


def test_compute_scores_applies_rotation_map():
    groups, images, _ = _build_groups([0.0] * 4, filters=["IR"])
    rotation_map = {m["stem"]: (3.0 if i >= 2 else 0.0) for i, (_, m) in enumerate(groups["IR"])}
    with _PatchedReadTif(images):
        scores = quality.compute_scores(groups, rotation_map=rotation_map)
        default = quality.compute_scores(groups)
    assert [r["rot_deg"] for r in scores["IR"]] == [0.0, 0.0, 3.0, 3.0]
    assert all(r["rot_deg"] == 0.0 for r in default["IR"])


# ── Real-data regression (skip silently if absent) ──────────────────────────

def test_real_saturn_271026_step():
    d = Path("D:/Astro/Sat/271026/step02_lucky_stack")
    if not d.is_dir():
        return
    groups = image_io.group_by_filter(d, target="Sat")
    if len(groups.get("IR", [])) < 10:
        return
    rot_map = detect_orientation_rotation_map(groups)
    late = [rot_map[m["stem"]] for _, m in groups["IR"] if m["timestamp"].hour >= 14]
    early = [rot_map[m["stem"]] for _, m in groups["IR"] if m["timestamp"].hour < 14]
    assert all(v == 0.0 for v in early), "pre-gap segment is the reference"
    assert all(abs(v - 5.1) < 0.5 for v in late), f"expected ≈ +5.1° after the gap, got {sorted(set(late))}"


def test_real_jupiter_260407_no_step():
    d = Path("D:/Astro/Mono_Temp_260407/bak/step02_lucky_stack")
    if not d.is_dir():
        return
    groups = image_io.group_by_filter(d, target="Jup")
    if len(groups.get("IR", [])) < 10:
        return
    assert detect_orientation_rotation_map(groups) == {}


if __name__ == "__main__":
    tests = [
        test_pairwise_estimate_recovers_known_rotation,
        test_single_step_is_detected_and_corrected,
        test_reference_is_longest_segment,
        test_flip_plus_residual_rotation,
        test_no_false_positive_without_step,
        test_continuous_drift_is_not_corrected,
        test_read_tif_oriented_rotation,
        test_compute_scores_applies_rotation_map,
        test_real_saturn_271026_step,
        test_real_jupiter_260407_no_step,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print("\nAll checks passed.")
