"""Regression test for meridian-flip detection / orientation normalisation.

Covers the 2026-10-21 Saturn finding: a meridian flip rotated the camera by
180° mid-session, and every step after Step 3 silently assumed one
orientation (session-wide pole_pa / derot_flip, windows mixing both sides).
pipeline/modules/orientation.py now detects the flip from the images and
rotates post-flip frames back before Step 3 scoring.

Synthetic frames: an orientation-asymmetric "planet" (disk + off-centre band
+ spot) with per-frame jitter and noise, written as FireCapture-named TIFs
so the real group_by_filter() path is exercised.

Run directly: python3 tests/test_orientation.py
Or via pytest: pytest tests/test_orientation.py -v
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.modules import image_io, orientation

try:
    import tifffile
except ImportError:  # pragma: no cover
    tifffile = None

SIZE = 200
FILTERS = ["IR", "R", "G"]


def _planet(seed: int, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
    """Disk with a band in the upper half and a spot lower-right: neither
    survives a 180° rotation, like Saturn's ring crossing / Jupiter's belts."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:SIZE, :SIZE].astype(np.float32)
    cx, cy = SIZE / 2 + dx, SIZE / 2 + dy
    rr = np.hypot(xx - cx, yy - cy)
    img = np.where(rr < 60, 0.6, 0.0).astype(np.float32)
    img[(rr < 60) & (np.abs(yy - (cy - 20)) < 6)] = 0.35
    img[np.hypot(xx - (cx + 25), yy - (cy + 30)) < 8] = 0.9
    img = cv2.GaussianBlur(img, (0, 0), 1.5)
    img += rng.normal(0, 0.01, img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


def _write_session(root: Path, flip_after: int | None, n: int = 10,
                   corrupt: tuple[str, int] | None = None) -> None:
    """n filter cycles, 10 min apart; frames from cycle > flip_after are
    rotated 180°. *corrupt* = (filter, cycle) replaces one frame by noise."""
    for k in range(n):
        for j, filt in enumerate(FILTERS):
            img = _planet(seed=100 * k + j, dx=(k % 3) - 1, dy=(k % 2) - 0.5)
            if flip_after is not None and k > flip_after:
                img = np.rot90(img, 2)
            if corrupt == (filt, k):
                img = np.random.default_rng(7).uniform(0, 0.2, img.shape).astype(np.float32)
            minute = 10 * k + 2 * j
            name = f"2026-10-21-{10 + minute // 60:02d}{minute % 60:02d}_0-U-{filt}-Sat_lucky.tif"
            image_io.write_tif_16bit(np.ascontiguousarray(img), root / name)


def test_detects_single_flip_in_all_filters():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_session(root, flip_after=5)
        groups = image_io.group_by_filter(root, "Sat")
        events = orientation.detect_flip_events(groups)
        assert len(events) == 1, events
        e = events[0]
        assert e["support"] == sorted(FILTERS)
        # Flip happened between cycle 5 (10:50-10:54) and cycle 6 (11:00).
        assert e["after"].strftime("%H:%M") == "10:54"
        assert e["before"].strftime("%H:%M") == "11:00"
        for _, entries in groups.items():
            for i, (_, meta) in enumerate(entries):
                assert orientation.is_flipped(meta["timestamp"], events) == (i > 5)


def test_no_flip_and_corrupt_frame_are_ignored():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_session(root, flip_after=None, corrupt=("R", 4))
        groups = image_io.group_by_filter(root, "Sat")
        assert orientation.detect_flip_events(groups) == []


def test_normalise_rotates_copies_and_leaves_inputs_untouched():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "in"
        work = Path(tmp) / "oriented"
        root.mkdir()
        _write_session(root, flip_after=5)
        before = {p.name: image_io.read_tif(p) for p in root.glob("*.tif")}

        groups = image_io.group_by_filter(root, "Sat")
        out = orientation.normalise_orientation(groups, work, report_dir=Path(tmp))

        assert (Path(tmp) / "orientation.json").exists()
        for p in root.glob("*.tif"):
            assert np.array_equal(image_io.read_tif(p), before[p.name])

        for filt, entries in out.items():
            ref = image_io.read_tif(entries[0][0])
            for i, (path, _) in enumerate(entries):
                assert (path.parent == work) == (i > 5), path
                img = image_io.read_tif(path)
                # Every output frame now matches the first frame's orientation.
                direct = np.corrcoef(ref.ravel(), img.ravel())[0, 1]
                rot = np.corrcoef(ref.ravel(), np.rot90(img, 2).ravel())[0, 1]
                assert direct > rot, (filt, i, direct, rot)
                if tifffile is not None:
                    assert tifffile.imread(str(path)).dtype == np.uint16

        # A re-run with no flip clears stale copies from the previous run.
        groups2 = {f: [(p, m) for p, m in e[:5]] for f, e in groups.items()}
        orientation.normalise_orientation(groups2, work)
        assert not work.exists()


if __name__ == "__main__":
    test_detects_single_flip_in_all_filters()
    test_no_flip_and_corrupt_frame_are_ignored()
    test_normalise_rotates_copies_and_leaves_inputs_untouched()
    print("all orientation tests passed")
