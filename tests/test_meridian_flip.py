"""Regression tests for discrete 180°-rotation (meridian flip / alt-az
re-seat) detection and correction.

Context: a GEM meridian flip rotates the camera image exactly 180°
(point-symmetric) at a discrete moment mid-session. Confirmed on real data
(Sat_TIFs/2026-09-19/, IR filter, 47 frames): frames 40-46 are exactly
180°-rotated relative to frames 0-39. The pairwise adjacent-frame NCC margin
(NCC(a, rot180(b)) - NCC(a, b)) was an order of magnitude clear of every
non-flip adjacent pair measured across six real no-flip Jupiter sessions
(+0.0068 at the real flip vs. -0.0097..-0.0204 for every non-flip pair) --
these tests guard that the detector (detect_orientation_flip_map) picks up
on exactly this signal, never on timestamp gaps, and that the fix is a
strict no-op for any frame/session it doesn't flag.

Run directly: python3 tests/test_meridian_flip.py
Or via pytest: pytest tests/test_meridian_flip.py -v
"""
from __future__ import annotations

import glob
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.modules import image_io, quality
from pipeline.modules.derotation import (
    _pairwise_rot180_ncc_margin,
    detect_orientation_flip_map,
)

H, W = 300, 300
CX, CY, DISK_R = 150.0, 150.0, 120.0


def make_belted_disk(seed: int = 0, noise_amp: float = 0.02) -> np.ndarray:
    """Asymmetric belted disk (horizontal belts + per-frame speckle noise) —
    NOT point-symmetric on average, so a 180° rotation is actually
    detectable (unlike a pure-speckle disk, which is close to point-
    symmetric in expectation and would under-test the asymmetry signal)."""
    yy_row = np.arange(H)[:, None]
    img = 0.5 + 0.35 * np.sin(yy_row / 18.0) + 0.1 * np.sin(yy_row / 55.0 + 1.0)
    img = np.broadcast_to(img, (H, W)).astype(np.float32).copy()
    yy, xx = np.mgrid[0:H, 0:W]
    mask = (xx - CX) ** 2 + (yy - CY) ** 2 < DISK_R ** 2
    rng = np.random.default_rng(seed)
    img = img + noise_amp * rng.standard_normal((H, W))
    img = (np.clip(img, 0.0, 1.0) * mask).astype(np.float32)
    return img


def test_pairwise_margin_sign_on_synthetic_belted_pair():
    """A true 180°-rotated pair must score margin > 0; an unrotated
    (noise-only-varying) pair must score margin < 0."""
    frame_a = make_belted_disk(seed=1)
    frame_b_same = make_belted_disk(seed=2)             # same orientation, different noise
    frame_b_rotated = np.rot90(make_belted_disk(seed=3), 2)  # genuinely flipped

    margin_same = _pairwise_rot180_ncc_margin(frame_a, frame_b_same)
    margin_rotated = _pairwise_rot180_ncc_margin(frame_a, frame_b_rotated)

    assert margin_same is not None and margin_rotated is not None
    assert margin_same < 0.0, f"unrotated pair should score margin<0, got {margin_same}"
    assert margin_rotated > 0.0, f"rotated pair should score margin>0, got {margin_rotated}"


def _build_synthetic_groups(
    n_frames: int,
    flip_from: "int | None",
    filters: List[str] = ["IR", "R"],
    cycle_sec: float = 60.0,
) -> Tuple[Dict[str, List[Tuple[Path, dict]]], List[bool]]:
    """Build a synthetic multi-filter 'groups' dict (image_io.group_by_filter
    shape) with real timestamps, optionally inserting exactly one 180°
    rotation event at frame index *flip_from* (that frame and every frame
    after it, across every filter, by real time). Returns (groups,
    ground_truth_flip_per_anchor_frame) for the first filter in *filters*
    (the expected anchor given _FILT_PREF_EXT_LOCAL's IR-first order).

    Images are returned as (path-like, meta) tuples where "path" is a
    plain object carrying the pre-built ndarray directly (monkeypatched
    read, see _patch_read_tif) rather than real files on disk -- avoids a
    temp-dir+TIF round trip for a pure in-memory detector test.
    """
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    groups: Dict[str, List[Tuple[Path, dict]]] = {f: [] for f in filters}
    truth: List[bool] = []
    images: Dict[str, np.ndarray] = {}

    # A real meridian flip takes real slew time, so there's necessarily a
    # capture gap around it -- no frame of any filter is captured near the
    # exact midpoint. Model that here (a large extra offset applied to
    # every frame from flip_from onward) so the synthetic timestamps don't
    # produce an unrealistic exact tie at the detector's boundary.
    flip_gap_sec = 600.0

    for i in range(n_frames):
        flipped = flip_from is not None and i >= flip_from
        truth.append(flipped)
        extra = flip_gap_sec if flipped else 0.0
        for j, filt in enumerate(filters):
            ts = t0 + timedelta(seconds=i * cycle_sec + j * (cycle_sec / len(filters)) + extra)
            stem = f"frame_{filt}_{i:03d}"
            path = Path(f"/synthetic/{stem}.tif")
            img = make_belted_disk(seed=1000 * i + j)
            if flipped:
                img = np.rot90(img, 2)
            images[str(path)] = img
            groups[filt].append((path, {"stem": stem, "timestamp": ts,
                                          "filter": filt, "target": "Jup"}))
    return groups, truth, images


class _PatchedReadTif:
    """Context manager: monkeypatch image_io.read_tif to serve pre-built
    in-memory arrays keyed by str(path), for synthetic-session tests."""

    def __init__(self, images: Dict[str, np.ndarray]):
        self._images = images
        self._orig = image_io.read_tif

    def __enter__(self):
        def _fake_read_tif(path):
            key = str(path)
            if key in self._images:
                return self._images[key].copy()
            return self._orig(path)
        image_io.read_tif = _fake_read_tif
        return self

    def __exit__(self, *exc):
        image_io.read_tif = self._orig


def test_detect_orientation_flip_map_synthetic_single_flip():
    """One inserted 180° flip must be detected and correctly segment BOTH
    filters by real time (cross-filter time-based propagation)."""
    groups, truth, images = _build_synthetic_groups(n_frames=20, flip_from=12)
    with _PatchedReadTif(images):
        flip_map = detect_orientation_flip_map(groups)

    assert flip_map, "expected a non-empty flip map for a session with an inserted flip"
    for filt in ("IR", "R"):
        for (path, meta), expected in zip(groups[filt], truth):
            stem = meta["stem"]
            assert flip_map.get(stem, False) == expected, (
                f"{stem}: expected flip180={expected}, got {flip_map.get(stem, False)}"
            )


def test_non_anchor_frames_near_transition_resolved_by_image():
    """Flip with a short slew gap right after a B frame (just before the next
    IR): the time midpoint between the bracketing IR frames falls BEFORE
    that B frame, so the time-only estimate would mark it flipped. The
    per-filter check against the previous same-filter frame must keep it
    unflipped, and every other frame must still match ground truth."""
    filters = ["IR", "R", "G", "B"]
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    groups: Dict[str, List[Tuple[Path, dict]]] = {f: [] for f in filters}
    images: Dict[str, np.ndarray] = {}
    truth: Dict[str, bool] = {}
    n_cycles, flip_after = 10, 5 * 4 + 3   # flip right after B5
    for k in range(n_cycles * 4):
        cyc, filt = divmod(k, 4)
        filt = filters[filt]
        flipped = k > flip_after
        ts = t0 + timedelta(seconds=k * 60 + (60 if flipped else 0))
        stem = f"frame_{filt}_{cyc:03d}"
        path = Path(f"/synthetic/{stem}.tif")
        img = make_belted_disk(seed=7000 + k)
        images[str(path)] = np.rot90(img, 2) if flipped else img
        groups[filt].append((path, {"stem": stem, "timestamp": ts,
                                      "filter": filt, "target": "Jup"}))
        truth[stem] = flipped

    with _PatchedReadTif(images):
        flip_map = detect_orientation_flip_map(groups)

    wrong = [s for s, exp in truth.items() if flip_map.get(s, False) != exp]
    assert not wrong, f"misassigned orientation: {wrong}"


def test_detect_orientation_flip_map_no_false_positive_on_pure_noise():
    """A session with no actual flip must produce an empty (or all-False)
    flip map -- the direct 'no threshold-tuning risk' regression guard."""
    groups, truth, images = _build_synthetic_groups(n_frames=20, flip_from=None)
    assert not any(truth)
    with _PatchedReadTif(images):
        flip_map = detect_orientation_flip_map(groups)

    assert all(not v for v in flip_map.values()), (
        f"expected no flips detected, got {sum(flip_map.values())} flipped frame(s)"
    )


def test_debounce_rejects_single_noisy_pair():
    """A single artificially-corrupted margin (simulating sensor noise) at
    one pair, with every neighboring pair behaving normally, must NOT
    commit a transition -- validates the skip-one confirmation/debounce."""
    import pipeline.modules.derotation as derotation_mod

    groups, truth, images = _build_synthetic_groups(n_frames=20, flip_from=None)

    orig_margin_fn = derotation_mod._pairwise_rot180_ncc_margin
    # Force exactly one pair (index 5) to report a spurious strong positive
    # margin, as if a single noisy frame pair looked like a flip.
    call_count = {"n": 0}

    def _spiked_margin(img_a, img_b, crop_margin_factor=1.3):
        call_count["n"] += 1
        if call_count["n"] == 6:  # the 6th pairwise call = index 5
            return 0.05
        return orig_margin_fn(img_a, img_b, crop_margin_factor)

    derotation_mod._pairwise_rot180_ncc_margin = _spiked_margin
    try:
        with _PatchedReadTif(images):
            flip_map = detect_orientation_flip_map(groups)
    finally:
        derotation_mod._pairwise_rot180_ncc_margin = orig_margin_fn

    assert all(not v for v in flip_map.values()), (
        "a single spiked pair must be rejected by the skip-one debounce, "
        f"got {sum(flip_map.values())} flipped frame(s)"
    )


def test_read_tif_oriented_is_exact_180(tmp_path=None):
    """read_tif_oriented(path, True) must equal an exact 180° rotation of
    read_tif(path); flip180=False must return a byte-identical array."""
    import tempfile
    img = make_belted_disk(seed=42)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "synthetic.tif"
        image_io.write_tif_16bit(img, p)

        base = image_io.read_tif(p)
        unflipped = image_io.read_tif_oriented(p, False)
        flipped = image_io.read_tif_oriented(p, True)

        np.testing.assert_array_equal(unflipped, base)
        np.testing.assert_array_equal(flipped, base[::-1, ::-1])


def test_compute_scores_flip_map_default_is_backward_compatible():
    """compute_scores(groups) with no flip_map argument must not error and
    every row must carry flip180=False -- protects the existing caller
    (quality_assess.py) and any future direct test callers."""
    groups, truth, images = _build_synthetic_groups(n_frames=4, flip_from=None, filters=["IR"])
    with _PatchedReadTif(images):
        scores = quality.compute_scores(groups)
    assert scores["IR"], "expected scored rows"
    for row in scores["IR"]:
        assert row["flip180"] is False


def test_compute_scores_applies_flip_map():
    """compute_scores(groups, flip_map=...) must tag rows with flip180
    according to the supplied map, keyed by stem."""
    groups, truth, images = _build_synthetic_groups(n_frames=6, flip_from=3, filters=["IR"])
    flip_map = {meta["stem"]: flipped for (path, meta), flipped in zip(groups["IR"], truth)}
    with _PatchedReadTif(images):
        scores = quality.compute_scores(groups, flip_map=flip_map)
    for row, expected in zip(scores["IR"], truth):
        assert row["flip180"] == expected


# ── Real-data regression (glob-guarded: skip silently if absent) ───────────

def test_real_saturn_session_detects_known_flip():
    """Sat_TIFs/2026-09-19/ (IR, 47 frames): frames 40-46 are confirmed
    (via manual NCC validation during this feature's design) to be exactly
    180°-rotated relative to frames 0-39. Must detect exactly one
    transition, with a flipped-frame count in a tolerant range around the
    confirmed ground truth (7 of 47)."""
    files = sorted(glob.glob("Sat_TIFs/2026-09-19/*-IR-*.tif"))
    if len(files) < 10:
        return  # session data not present in this environment — skip

    groups = image_io.group_by_filter(Path("Sat_TIFs/2026-09-19"), target="Sat")
    if not groups.get("IR"):
        return

    flip_map = detect_orientation_flip_map(groups)
    n_flipped = sum(1 for p, m in groups["IR"] if flip_map.get(m["stem"], False))
    assert 5 <= n_flipped <= 9, (
        f"expected ~7/{len(groups['IR'])} IR frames flipped (confirmed ground truth), got {n_flipped}"
    )


def test_real_no_flip_sessions_produce_empty_flip_map():
    """Direct automated proof of the 'zero changed output for existing
    sessions' no-op guarantee: every known no-flip Jupiter session under
    TIFs/ must produce an empty/all-False flip map."""
    found_any = False
    for day in ("260321", "260327", "260328", "260417", "260424", "260505"):
        tif_dir = Path("TIFs") / day / "TIFs"
        if not tif_dir.is_dir():
            continue
        groups = image_io.group_by_filter(tif_dir, target="Jup")
        if not groups.get("IR"):
            continue
        found_any = True
        flip_map = detect_orientation_flip_map(groups)
        n_flipped = sum(1 for v in flip_map.values() if v)
        assert n_flipped == 0, f"{day}: expected no flips, got {n_flipped}"
    if not found_any:
        return  # no session data present in this environment — skip


if __name__ == "__main__":
    tests = [
        test_pairwise_margin_sign_on_synthetic_belted_pair,
        test_detect_orientation_flip_map_synthetic_single_flip,
        test_non_anchor_frames_near_transition_resolved_by_image,
        test_detect_orientation_flip_map_no_false_positive_on_pure_noise,
        test_debounce_rejects_single_noisy_pair,
        test_read_tif_oriented_is_exact_180,
        test_compute_scores_flip_map_default_is_backward_compatible,
        test_compute_scores_applies_flip_map,
        test_real_saturn_session_detects_known_flip,
        test_real_no_flip_sessions_produce_empty_flip_map,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print("\nAll checks passed.")
