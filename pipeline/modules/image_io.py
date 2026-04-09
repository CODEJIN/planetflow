"""
Image I/O and filename utilities for the planetary imaging pipeline.

Filename convention (AutoStakkert!4 output via PIPP pre-processing):
    YYYY-MM-DD-HHMM_D-U-FILTER-TARGET_pipp_lapl3_ap51.tif

    HHMM_D  → HH:MM UTC, D = tenths of a minute (D × 6 seconds)
    FILTER  → IR | R | G | B | CH4
    TARGET  → Jup | Sat | Mar | …
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Optional: tifffile gives better 16/32-bit TIF support
try:
    import tifffile
    _HAS_TIFFFILE = True
except ImportError:
    _HAS_TIFFFILE = False


# ── Filename parsing ───────────────────────────────────────────────────────────

# Matches: 2026-03-20-1046_1-U-IR-Jup_pipp…
_FNAME_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})"          # group 1: date
    r"-(\d{4}_\d)"                    # group 2: HHMM_D
    r"-[^-]+"                         # camera id (ignored)
    r"-([A-Za-z0-9]+)"                # group 3: filter
    r"-([A-Za-z]+)"                   # group 4: target
    r"_"
)


def parse_filename(path: Path) -> Optional[Dict]:
    """Extract date, UTC timestamp, filter, and target from an AS!4 TIF filename.

    Returns None if the filename does not match the expected pattern.
    """
    m = _FNAME_RE.match(path.name)
    if not m:
        return None

    date_str, time_str, filter_name, target = m.groups()

    # time_str = "HHMM_D"  e.g. "1046_1"
    hh = int(time_str[0:2])
    mm = int(time_str[2:4])
    d  = int(time_str[5])        # tenths of a minute
    total_seconds = hh * 3600 + mm * 60 + d * 6

    date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    timestamp = date + timedelta(seconds=total_seconds)

    return {
        "date": date_str,
        "timestamp": timestamp,
        "filter": filter_name,
        "target": target,
        "stem": path.stem,
    }


def group_by_filter(
    tif_dir: Path,
    target: str = "Jup",
) -> Dict[str, List[Tuple[Path, Dict]]]:
    """Scan *tif_dir* for TIF files and group them by filter name.

    Returns:
        {filter_name: [(path, meta), ...]}  sorted by timestamp within each group.
    """
    groups: Dict[str, List[Tuple[Path, Dict]]] = {}

    for p in sorted(tif_dir.glob("*.tif")):
        meta = parse_filename(p)
        if meta is None or meta["target"] != target:
            continue
        f = meta["filter"]
        groups.setdefault(f, []).append((p, meta))

    for f in groups:
        groups[f].sort(key=lambda x: x[1]["timestamp"])

    return groups


# ── Image reading ──────────────────────────────────────────────────────────────

def read_tif(path: Path) -> np.ndarray:
    """Read a TIF file and return a float32 array normalised to [0, 1].

    Supports 8-bit and 16-bit grayscale TIFs.
    Uses tifffile when available (better multi-page / float TIF support),
    falls back to OpenCV otherwise.
    """
    if _HAS_TIFFFILE:
        img = tifffile.imread(str(path))
    else:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")

    img = img.astype(np.float32)

    # Normalise to [0, 1] based on bit depth
    if img.max() > 1.0:
        max_val = 65535.0 if img.max() > 255 else 255.0
        img /= max_val

    return img


# ── Image writing ──────────────────────────────────────────────────────────────

def write_png_16bit(image: np.ndarray, path: Path) -> None:
    """Write a float [0, 1] image as a 16-bit grayscale PNG."""
    arr = np.clip(image * 65535.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(str(path), arr)


def write_png_autostretch(
    image: np.ndarray,
    path: Path,
    plow: float = 0.5,
    phigh: float = 99.5,
) -> None:
    """Write a float [0, 1] image as a 16-bit PNG with percentile auto-stretch.

    Stretching makes the preview visible for quality inspection even when the
    raw pixel values occupy only a small portion of the dynamic range.
    """
    lo, hi = np.percentile(image, [plow, phigh])
    span = hi - lo
    if span < 1e-9:
        stretched = np.zeros_like(image)
    else:
        stretched = np.clip((image - lo) / span, 0.0, 1.0)
    write_png_16bit(stretched, path)


def write_tif_16bit(image: np.ndarray, path: Path) -> None:
    """Write a float [0, 1] image as a 16-bit grayscale TIF."""
    arr = np.clip(image * 65535.0, 0, 65535).astype(np.uint16)
    if _HAS_TIFFFILE:
        tifffile.imwrite(str(path), arr)
    else:
        cv2.imwrite(str(path), arr)


def read_png(path: Path) -> np.ndarray:
    """Read a PNG (8-bit or 16-bit, grayscale or colour) and return float32 [0, 1].

    Colour images are returned as (H, W, 3) in RGB order.
    Grayscale images are returned as (H, W).
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    img = img.astype(np.float32)
    max_val = 65535.0 if img.max() > 255 else 255.0
    img /= max_val
    # OpenCV reads colour as BGR → convert to RGB
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def write_png_color_16bit(image: np.ndarray, path: Path) -> None:
    """Write a float [0, 1] (H, W, 3) RGB image as a 16-bit colour PNG."""
    arr = np.clip(image * 65535.0, 0, 65535).astype(np.uint16)
    # OpenCV expects BGR
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def write_tif_color_16bit(image: np.ndarray, path: Path) -> None:
    """Write a float [0, 1] (H, W, 3) RGB image as a 16-bit colour TIF."""
    arr = np.clip(image * 65535.0, 0, 65535).astype(np.uint16)
    if _HAS_TIFFFILE:
        # tifffile preserves channel order (RGB) natively
        tifffile.imwrite(str(path), arr)
    else:
        # OpenCV expects BGR
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)
