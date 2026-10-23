"""
Meridian-flip detection and orientation normalisation for Step 3 inputs.

A German-equatorial mount's meridian flip rotates the camera by exactly 180°
relative to the sky partway through a session. Everything downstream of
Step 3 assumes ONE image orientation per session: Step 4 measures pole_pa
and derot_flip once and applies them to every window, and windows freely mix
frames from either side of the flip. A flipped segment is therefore warped
in the wrong rotation direction, and windows straddling the flip stack
upside-down frames on top of each other (confirmed on 2026-10-21 Saturn
data: flip at 14:04-14:10 UT, all 5 filters).

The fix lives at the Step 3 entry point because that is where both input
paths meet -- Step 2 output and externally stacked TIFs (AutoStakkert etc.)
are both read from config.input_dir. Detection is image-based (no capture
metadata needed), so it also works on external stacks.

Detection: per filter, consecutive frames are compared directly and with
the later frame rotated 180° (high-pass filtered, centred, sub-pixel aligned
by phase correlation). A pair where the rotated match clearly wins is a flip
candidate; it must then hold across the neighbouring frames too (so one bad
frame can't fake a flip), and a strict majority of the filters that have
frames on both sides of the flip must agree (one real flip happens at the
same moment in every filter).

Normalisation: frames after an odd number of flips are rotated 180° back to
the session's FIRST orientation. Rotated copies are written to a working
folder and substituted into the file groups -- the input files themselves
(possibly the user's external stacks) are never modified.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from pipeline.modules import image_io

try:
    import tifffile
    _HAS_TIFFFILE = True
except ImportError:
    _HAS_TIFFFILE = False


# Minimum (NCC_rotated - NCC_direct) for a pair to count as a flip. Real
# flips measured 0.15-0.42 across IR/R/G/B/CH4 (Saturn 2026-10-21 and
# 2026-09-19); non-flip pairs sit far below zero except for corrupted
# frames, where both NCCs collapse together and the difference stays ~0.
_FLIP_MARGIN = 0.05

# Frames on each side of a candidate pair used to confirm it.
_CONFIRM_SPAN = 2

# Gaussian sigma (px) of the low-pass removed before matching. Keeps belt /
# ring / shadow structure (orientation-asymmetric) and drops the smooth disk
# profile, which is nearly 180°-symmetric and would dominate the NCC.
_HIGHPASS_SIGMA = 3.0


# ── Image comparison ───────────────────────────────────────────────────────────

def _prep(path: Path) -> np.ndarray:
    """Load a frame as a centred, high-passed, Hann-windowed luminance image.

    The planet is translated so its brightness centroid sits exactly at the
    array centre; np.rot90(img, 2) then rotates it about the planet centre.
    """
    img = image_io.read_tif(path)
    lum = img.mean(axis=2) if img.ndim == 3 else img
    lum = np.clip(lum.astype(np.float32) - float(np.median(lum)), 0.0, None)

    h, w = lum.shape
    peak = float(np.percentile(lum, 99.5))
    ys, xs = np.nonzero(lum > 0.2 * peak) if peak > 0 else ((), ())
    if len(xs) > 0:
        M = np.float32([[1, 0, (w - 1) / 2 - xs.mean()],
                        [0, 1, (h - 1) / 2 - ys.mean()]])
        lum = cv2.warpAffine(lum, M, (w, h), flags=cv2.INTER_LINEAR)

    hp = lum - cv2.GaussianBlur(lum, (0, 0), _HIGHPASS_SIGMA)
    return hp * cv2.createHanningWindow((w, h), cv2.CV_32F)


def _aligned_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """NCC of *a* and *b* after phase-correlation alignment of *b* onto *a*."""
    if a.shape != b.shape:
        return float("nan")
    (dx, dy), _ = cv2.phaseCorrelate(a.astype(np.float64), b.astype(np.float64))
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    b = cv2.warpAffine(b, M, (b.shape[1], b.shape[0]), flags=cv2.INTER_LINEAR)
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = float(np.sqrt((a0 * a0).sum() * (b0 * b0).sum()))
    return float((a0 * b0).sum() / denom) if denom > 0 else float("nan")


def flip_vote(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    """Return (vote, ncc_direct, ncc_rot180) for prepped frames *a*, *b*.

    vote = ncc_rot180 - ncc_direct; positive means *b* is upside down
    relative to *a*. NaN NCCs (blank / mismatched frames) give vote 0.
    """
    direct = _aligned_ncc(a, b)
    rot = _aligned_ncc(a, np.ascontiguousarray(np.rot90(b, 2)))
    vote = rot - direct
    return (0.0 if not np.isfinite(vote) else float(vote)), direct, rot


# ── Detection ──────────────────────────────────────────────────────────────────

def _filter_candidates(
    entries: List[Tuple[Path, dict]],
    cancel_event=None,
) -> List[dict]:
    """Confirmed flip candidates for one filter's time-sorted frames."""
    n = len(entries)
    if n < 2:
        return []

    # Pass 1: consecutive-pair votes, streaming (only two frames in memory).
    votes: List[float] = []
    prev = _prep(entries[0][0])
    for i in range(1, n):
        if cancel_event is not None and cancel_event.is_set():
            return []
        cur = _prep(entries[i][0])
        votes.append(flip_vote(prev, cur)[0])
        prev = cur

    # Pass 2: confirm each candidate against its neighbours. Cross pairs
    # (before-side frame, after-side frame) must also prefer the rotated
    # match -- a single corrupt frame only spoils the pairs it is part of,
    # so the median rejects it.
    out: List[dict] = []
    for i, v in enumerate(votes):
        if v <= _FLIP_MARGIN:
            continue
        before = range(max(0, i - _CONFIRM_SPAN + 1), i + 1)
        after = range(i + 1, min(n, i + 1 + _CONFIRM_SPAN))
        preps = {k: _prep(entries[k][0]) for k in (*before, *after)}
        cross = [flip_vote(preps[j], preps[k])[0] for j in before for k in after]
        confirm = float(np.median(cross))
        if confirm > _FLIP_MARGIN:
            out.append({
                "t0": entries[i][1]["timestamp"],
                "t1": entries[i + 1][1]["timestamp"],
                "vote": v,
                "confirm": confirm,
                "stem0": entries[i][1]["stem"],
                "stem1": entries[i + 1][1]["stem"],
            })
    return out


def detect_flip_events(
    groups: Dict[str, List[Tuple[Path, dict]]],
    cancel_event=None,
) -> List[dict]:
    """Find meridian-flip events across all filters.

    Returns a time-sorted list of accepted events:
        {"time", "after", "before", "support", "eligible", "candidates"}
    where the flip happened between "after" and "before" (the overlap of
    every supporting filter's straddling pair) and "time" is its midpoint.
    """
    candidates: List[dict] = []
    for filt in sorted(groups):
        for c in _filter_candidates(groups[filt], cancel_event):
            candidates.append({**c, "filter": filt})
    if not candidates:
        return []

    # Cluster candidates whose straddling intervals overlap: one physical
    # flip lies inside every filter's straddling pair, so their intervals
    # share a common sub-interval.
    candidates.sort(key=lambda c: c["t0"])
    clusters: List[dict] = []
    for c in candidates:
        cl = clusters[-1] if clusters else None
        if cl is not None and c["t0"] < cl["hi"] and c["t1"] > cl["lo"]:
            cl["lo"] = max(cl["lo"], c["t0"])
            cl["hi"] = min(cl["hi"], c["t1"])
            cl["members"].append(c)
        else:
            clusters.append({"lo": c["t0"], "hi": c["t1"], "members": [c]})

    events: List[dict] = []
    for cl in clusters:
        lo, hi = cl["lo"], cl["hi"]
        support = sorted({m["filter"] for m in cl["members"]})
        # Filters with frames on both sides of the flip could have seen it.
        eligible = sorted(
            f for f, entries in groups.items()
            if any(m["timestamp"] <= lo for _, m in entries)
            and any(m["timestamp"] >= hi for _, m in entries)
        )
        accepted = 2 * len(support) > len(eligible)
        print(
            f"  [orient] flip candidate {lo:%H:%M:%S}–{hi:%H:%M:%S} UT: "
            f"{len(support)}/{len(eligible)} filter(s) agree {support} → "
            f"{'ACCEPTED' if accepted else 'rejected'}"
        )
        if accepted:
            events.append({
                "time": lo + (hi - lo) / 2,
                "after": lo,
                "before": hi,
                "support": support,
                "eligible": eligible,
                "candidates": cl["members"],
            })
    return events


def is_flipped(timestamp: datetime, events: List[dict]) -> bool:
    """True when *timestamp* follows an odd number of flip events."""
    return sum(1 for e in events if e["time"] < timestamp) % 2 == 1


# ── Normalisation ──────────────────────────────────────────────────────────────

def _write_rot180_copy(src: Path, dst: Path) -> None:
    """Write *src* rotated by 180° to *dst*, preserving dtype and channels."""
    if _HAS_TIFFFILE:
        arr = tifffile.imread(str(src))
        tifffile.imwrite(str(dst), np.ascontiguousarray(np.rot90(arr, 2, axes=(0, 1))))
    else:
        arr = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(f"Cannot read: {src}")
        cv2.imwrite(str(dst), np.ascontiguousarray(np.rot90(arr, 2, axes=(0, 1))))


def normalise_orientation(
    groups: Dict[str, List[Tuple[Path, dict]]],
    work_dir: Path,
    report_dir: Optional[Path] = None,
    cancel_event=None,
) -> Dict[str, List[Tuple[Path, dict]]]:
    """Detect meridian flips and return *groups* in one consistent orientation.

    Frames after an odd number of flips are rotated 180° (back to the
    orientation of the session's first frames). Rotated copies go to
    *work_dir* -- which this function owns and clears on every call so no
    stale copies from an earlier run linger -- and replace the original
    paths in the returned groups. Input files are never modified. When
    *report_dir* is given, orientation.json is written there.
    """
    print("  [orient] Checking for meridian flips (180° image rotation)…")
    if work_dir.exists():
        shutil.rmtree(work_dir)

    events = detect_flip_events(groups, cancel_event)
    rotated: List[dict] = []
    out_groups = groups

    if events:
        work_dir.mkdir(parents=True, exist_ok=True)
        out_groups = {}
        for filt, entries in groups.items():
            new_entries = []
            for path, meta in entries:
                if is_flipped(meta["timestamp"], events):
                    dst = work_dir / path.name
                    _write_rot180_copy(path, dst)
                    rotated.append({"filter": filt, "stem": meta["stem"],
                                    "source": str(path), "oriented": str(dst)})
                    path = dst
                new_entries.append((path, meta))
            out_groups[filt] = new_entries
        print(f"  [orient] {len(events)} flip(s) found; rotated {len(rotated)} "
              f"frame(s) by 180° → {work_dir}")
    else:
        print("  [orient] No meridian flip detected — all frames used as-is")

    if report_dir is not None:
        report = {
            "events": [
                {
                    "time": e["time"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "between": [e["after"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                                e["before"].strftime("%Y-%m-%dT%H:%M:%SZ")],
                    "support": e["support"],
                    "eligible": e["eligible"],
                    "pairs": [
                        {"filter": c["filter"], "from": c["stem0"], "to": c["stem1"],
                         "vote": round(c["vote"], 4), "confirm": round(c["confirm"], 4)}
                        for c in e["candidates"]
                    ],
                }
                for e in events
            ],
            "rotated_frames": rotated,
        }
        with open(report_dir / "orientation.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("  → orientation.json")

    return out_groups
