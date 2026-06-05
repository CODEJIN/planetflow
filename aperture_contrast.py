"""
Aperture Contrast Measurement for Multi-rate De-rotation Evaluation

Measures the aperture contrast improvement produced by multi-rate stacking,
as described in Section 3.3 of the paper. Compares a baseline (planet-only
de-rotation) image against a composite (multi-rate stacked) image for each
satellite or shadow target.

Aperture contrast is defined as |P - B| / B, where:
  P  = peak signal in the ROI (max for satellite, min for shadow)
  B  = median of the local background annulus

Zones (in units of reference radius r = 10% of apparent disk diameter):
  ROI        0.0 – 1.0 × r   (signal)
  Gap        1.0 – 1.5 × r   (PSF wing buffer)
  Background 1.5 – 2.5 × r   (local background)

────────────────────────────────────────────────────────────────────────
Usage (auto — Step 3 input TIF folder, runs full pipeline internally):
    python aperture_contrast.py --tif-dir path/to/lucky_stacked_tifs/ --output-dir ./out

    --output-dir saves everything: step4 stacks (out/baseline/, out/composite/),
    derotation logs, and comparison PNGs — all in one place.
    Omitting --output-dir runs in a temp directory (results printed only, no files kept).

    Optional:
      --filter IR          Preferred filter for contrast measurement (default: IR)

Usage (semi-auto — existing Step 4 output directories):
    python aperture_contrast.py \
        --baseline-dir  path/to/step04_baseline/step04_derotated/ \
        --composite-dir path/to/step04_composite/step04_derotated/ \
        --output-dir    ./out

Usage (single pair, manual coordinates):
    python aperture_contrast.py \
        --baseline  path/to/baseline.tif \
        --composite path/to/composite.tif \
        --x 312 --y 205 --r 18.4 --type satellite

Usage (batch from CSV):
    python aperture_contrast.py --csv measurements.csv

CSV format (one row per measurement):
    baseline,composite,x_px,y_px,r_px,type,label
    win01/IR_baseline.tif,win01/IR_composite.tif,312,205,18.4,satellite,2026-03-28_IR_Io
    ...
    (type: "satellite" or "shadow")
    r_px = 10% of the satellite/shadow apparent disk diameter (NOT the apparent radius).
    e.g. apparent disk diameter 184 px → r_px = 18.4
    baseline = Step 4 output with satellite_composite_enabled=False (planet-only stack)
    composite = Step 4 output with satellite_composite_enabled=True
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional

import cv2
import numpy as np


# ── Core measurement ──────────────────────────────────────────────────────────

@dataclass
class ApertureContrastResult:
    label: str
    obj_type: Literal["satellite", "shadow"]
    x_px: float
    y_px: float
    r_px: float
    contrast_baseline: float
    contrast_composite: float
    filter_name: str = ""
    session: str = ""
    window: str = ""   # e.g. "window_01" (without session prefix)

    @property
    def delta_pct(self) -> float:
        if self.contrast_baseline == 0:
            return float("nan")
        return 100.0 * (self.contrast_composite - self.contrast_baseline) / self.contrast_baseline

    def __str__(self) -> str:
        return (
            f"{self.label:<35}  "
            f"baseline={self.contrast_baseline:.4f}  "
            f"composite={self.contrast_composite:.4f}  "
            f"Δ={self.delta_pct:+.1f}%"
        )


def _read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
    if img.ndim == 3:
        img = np.mean(img, axis=2)
    return img


def _annular_mask(
    shape: tuple[int, int],
    cx: float,
    cy: float,
    r_inner: float,
    r_outer: float,
) -> np.ndarray:
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    dist = np.hypot(xx - cx, yy - cy)
    return (dist >= r_inner) & (dist < r_outer)


_WAVELET_AMOUNTS_DISPLAY = [200.0, 200.0, 200.0, 0.0, 0.0, 0.0]


def _apply_wavelet_display(
    img: np.ndarray,
    disk_cx: float,
    disk_cy: float,
    disk_r: float,
) -> np.ndarray:
    """Apply wavelet sharpening for visual display only (not for contrast measurement)."""
    try:
        from pipeline.modules.wavelet import sharpen_disk_aware
        return sharpen_disk_aware(
            img, disk_cx, disk_cy, disk_r,
            levels=6,
            amounts=_WAVELET_AMOUNTS_DISPLAY,
        )
    except Exception:
        return img


def _save_comparison_image(
    result: "ApertureContrastResult",
    img_base: np.ndarray,
    img_comp: np.ndarray,
    save_dir: Path,
    disk_cx: float = 0.0,
    disk_cy: float = 0.0,
    disk_r: float = 0.0,
) -> None:
    """Save a side-by-side comparison PNG with zoomed ROI insets.

    Top row: full baseline | full composite (with zone circles overlaid).
    Bottom row: baseline ROI zoom | composite ROI zoom (5× zoom around aperture).
    Wavelet sharpening ([200,200,200,0,0,0]) applied to display images only.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D
    except ImportError:
        print("  [warning] matplotlib not available — skipping image output")
        return

    x, y, r = result.x_px, result.y_px, result.r_px
    h, w    = img_base.shape

    # Wavelet sharpening for display (disk_r > 0 means disk geometry is available)
    if disk_r > 0.0:
        disp_base = _apply_wavelet_display(img_base, disk_cx, disk_cy, disk_r)
        disp_comp = _apply_wavelet_display(img_comp, disk_cx, disk_cy, disk_r)
    else:
        disp_base, disp_comp = img_base, img_comp

    vmin = 0.0
    vmax = 1.0

    # Zoom crop: 5× r padding around the aperture center, clamped to image bounds
    pad   = int(np.ceil(5.0 * r))
    x0    = max(0, int(x) - pad)
    y0    = max(0, int(y) - pad)
    x1    = min(w, int(x) + pad)
    y1    = min(h, int(y) + pad)
    zx = x - x0  # aperture center in crop coords
    zy = y - y0

    aspect  = w / h
    fig_w   = min(20.0, max(14.0, aspect * 10.0))
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_w / aspect + 2.5),
                              facecolor="#111111",
                              gridspec_kw={"height_ratios": [3, 2]})
    fig.subplots_adjust(hspace=0.08, wspace=0.04,
                        left=0.01, right=0.99, top=0.93, bottom=0.02)
    fig.suptitle(
        f"Aperture Contrast  ·  {result.label}  ·  Δ = {result.delta_pct:+.1f}%  ({result.obj_type})",
        color="white", fontsize=10,
    )

    CIRCLE_KW = [
        dict(radius=1.0 * r, color="#F5A623", lw=1.5, ls="-",  label="ROI (0–1r)"),
        dict(radius=1.5 * r, color="#F5A623", lw=1.0, ls="--", label="Gap (1–1.5r)"),
        dict(radius=2.5 * r, color="#5B9BD5", lw=1.0, ls="--", label="Background (1.5–2.5r)"),
    ]

    zoom_base = disp_base[y0:y1, x0:x1]
    zoom_comp = disp_comp[y0:y1, x0:x1]

    panels = [
        (axes[0, 0], disp_base, x,  y,  "(A) without Multi-Stack", result.contrast_baseline),
        (axes[0, 1], disp_comp, x,  y,  "(B) with Multi-Stack",    result.contrast_composite),
        (axes[1, 0], zoom_base, zx, zy, "(A) ROI zoom",            result.contrast_baseline),
        (axes[1, 1], zoom_comp, zx, zy, "(B) ROI zoom",            result.contrast_composite),
    ]

    for i, (ax, img, cx_p, cy_p, title, contrast) in enumerate(panels):
        ax.set_facecolor("#111111")
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax,
                  origin="upper", interpolation="nearest")
        ax.set_title(title, color="white", fontsize=8.5, pad=3)

        for kw in CIRCLE_KW:
            circ = mpatches.Circle(
                (cx_p, cy_p), kw["radius"],
                fill=False, edgecolor=kw["color"],
                linewidth=kw["lw"], linestyle=kw["ls"],
            )
            ax.add_patch(circ)

        ax.plot(cx_p, cy_p, "+", color="white", ms=8, mew=1.5)

        p_label = "P=max" if result.obj_type == "satellite" else "P=min"
        ax.text(0.03, 0.97,
                f"C = {contrast:.4f}  ({p_label})",
                transform=ax.transAxes,
                color="white", fontsize=7.5, va="top",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="black", alpha=0.6, edgecolor="none"))
        ax.axis("off")

    # Legend on top-left panel only
    legend_handles = [
        Line2D([0], [0], color=kw["color"], lw=kw["lw"],
               ls=kw["ls"], label=kw["label"])
        for kw in CIRCLE_KW
    ]
    axes[0, 0].legend(
        handles=legend_handles, loc="upper right",
        fontsize=7, framealpha=0.65,
        facecolor="black", labelcolor="white", edgecolor="none",
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    safe  = result.label.replace("/", "_").replace(" ", "_")
    fpath = save_dir / f"{safe}.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    print(f"    → {fpath.name}")


def measure_aperture_contrast(
    img: np.ndarray,
    x_px: float,
    y_px: float,
    r_px: float,
    obj_type: Literal["satellite", "shadow"],
) -> float:
    """Return |P - B| / B for a single object in img."""
    roi_mask = _annular_mask(img.shape, x_px, y_px, 0.0, 1.0 * r_px)
    bg_mask  = _annular_mask(img.shape, x_px, y_px, 1.5 * r_px, 2.5 * r_px)

    roi_pixels = img[roi_mask]
    bg_pixels  = img[bg_mask]

    if roi_pixels.size == 0 or bg_pixels.size == 0:
        return float("nan")

    P = float(roi_pixels.max()) if obj_type == "satellite" else float(roi_pixels.min())
    B = float(np.median(bg_pixels))

    if B == 0:
        return float("nan")
    return abs(P - B) / B


def measure_pair(
    baseline_path: Path,
    composite_path: Path,
    x_px: float,
    y_px: float,
    r_px: float,
    obj_type: Literal["satellite", "shadow"],
    label: str = "",
) -> ApertureContrastResult:
    """Compare aperture contrast between a baseline and composite image."""
    base_img = _read_image(baseline_path)
    comp_img = _read_image(composite_path)

    c_base = measure_aperture_contrast(base_img, x_px, y_px, r_px, obj_type)
    c_comp = measure_aperture_contrast(comp_img, x_px, y_px, r_px, obj_type)

    return ApertureContrastResult(
        label=label or f"{baseline_path.stem}",
        obj_type=obj_type,
        x_px=x_px,
        y_px=y_px,
        r_px=r_px,
        contrast_baseline=c_base,
        contrast_composite=c_comp,
    )


# ── Window de-overlap (matches Step 9 / summary_grid logic) ──────────────────

def _deoverlap_windows(windows: list) -> list:
    """Greedy non-overlapping selection from step3 results_03['windows'].

    Windows are sorted by quality descending; each candidate is accepted only
    if its center_time is at least one window duration away from every already-
    accepted window.  Result is re-sorted chronologically.
    """
    if not windows:
        return []
    try:
        duration_sec = (
            windows[0]["window_end"] - windows[0]["window_start"]
        ).total_seconds()
    except Exception:
        duration_sec = 900.0

    sorted_wins = sorted(windows, key=lambda w: w.get("window_quality", 0.0), reverse=True)
    accepted: list = []
    accepted_times: list = []
    for win in sorted_wins:
        t = win["center_time"]
        if not any(abs((t - at).total_seconds()) < duration_sec for at in accepted_times):
            accepted.append(win)
            accepted_times.append(t)
    return sorted(accepted, key=lambda w: w["center_time"])


def _deoverlap_win_dirs(win_dirs: List[Path]) -> List[Path]:
    """Greedy non-overlapping selection from window_XX directories.

    Reads derotation_log.json from each directory to obtain center_time,
    window_quality, and window duration.
    """
    info = []
    for wd in win_dirs:
        log_path = wd / "derotation_log.json"
        if not log_path.exists():
            continue
        with open(log_path) as f:
            d = json.load(f)
        try:
            center  = datetime.strptime(d["center_time"], "%Y-%m-%dT%H:%M:%SZ")
            quality = float(d.get("window_quality", 0.0))
            start   = datetime.strptime(d["window_start"], "%Y-%m-%dT%H:%M:%SZ")
            end     = datetime.strptime(d["window_end"],   "%Y-%m-%dT%H:%M:%SZ")
            duration_sec = (end - start).total_seconds()
        except (KeyError, ValueError):
            continue
        info.append({"dir": wd, "center": center, "quality": quality,
                     "duration_sec": duration_sec})

    if not info:
        return []

    duration_sec = info[0]["duration_sec"]
    sorted_info  = sorted(info, key=lambda x: x["quality"], reverse=True)
    accepted_dirs: List[Path] = []
    accepted_times: list = []
    for item in sorted_info:
        t = item["center"]
        if not any(abs((t - at).total_seconds()) < duration_sec for at in accepted_times):
            accepted_dirs.append(item["dir"])
            accepted_times.append(t)
    return sorted(accepted_dirs, key=lambda d: d.name)


# ── Auto modes (pipeline integration) ────────────────────────────────────────

def _apparent_radius_px(moon_name: str, t_ref: datetime, plate_scale: float) -> float:
    """Compute apparent radius of a Galilean moon in pixels at observation time."""
    _RADII_KM = {"Io": 1821.6, "Europa": 1560.8, "Ganymede": 2634.1, "Callisto": 2410.3}
    try:
        from pipeline.modules.satellite_tracker import _load_skyfield_kernels, _MOON_SF_ID
        sf = _load_skyfield_kernels()
        if sf is None:
            raise RuntimeError("Skyfield unavailable")
        ts, eph, jup_moons = sf
        sf_id = _MOON_SF_ID.get(moon_name)
        if sf_id is None:
            raise KeyError(moon_name)
        t_sf  = ts.utc(t_ref.year, t_ref.month, t_ref.day,
                        t_ref.hour, t_ref.minute, t_ref.second)
        dist_au = eph["earth"].at(t_sf).observe(jup_moons[sf_id]).distance().au
        r_km    = _RADII_KM.get(moon_name, 1560.8)
        return r_km / (dist_au * 149_597_870.7) * 206_265.0 / plate_scale
    except Exception:
        r_km = _RADII_KM.get(moon_name, 1560.8)
        return r_km / (6.0 * 149_597_870.7) * 206_265.0 / plate_scale


def _filter_satellite_windows(windows: list, groups: dict) -> list:
    """Keep only windows where at least one satellite or shadow is on disk.

    Uses a reference TIF from the input folder (middle frame of the preferred
    filter) to estimate disk center, then queries SatelliteTracker for each
    window center time.
    """
    try:
        from pipeline.modules.satellite_tracker import SatelliteTracker
        from pipeline.modules.derotation import find_disk_center
        from pipeline.modules.image_io import read_tif
    except ImportError:
        return windows

    # Pick a reference frame for disk detection
    filt_order = ["IR", "R", "G", "B"] + list(groups.keys())
    ref_tif = None
    for f in filt_order:
        if f in groups and groups[f]:
            ref_tif = groups[f][len(groups[f]) // 2][0]
            break
    if ref_tif is None:
        return windows

    try:
        img  = read_tif(ref_tif)
        cx, cy, r, *_ = find_disk_center(img)
    except Exception:
        return windows

    tracker     = SatelliteTracker()
    t_sample    = windows[0]["center_time"]
    plate_scale = tracker.get_plate_scale(r, t_sample)

    selected = []
    for win in windows:
        t = win["center_time"]
        body_pos = tracker.get_positions([t], cx, cy, r, plate_scale)
        has_body = any(
            p is not None and p.on_disk
            for positions in body_pos.values()
            for p in positions
        )
        if has_body:
            selected.append(win)
            continue
        try:
            shad_pos = tracker.get_shadow_positions([t], cx, cy, r, plate_scale) or {}
            has_shad = any(
                p is not None and p.on_disk
                for positions in shad_pos.values()
                for p in positions
            )
            if has_shad:
                selected.append(win)
        except Exception:
            pass

    return selected


def _measure_window_pair(
    base_win_dir: Path,
    comp_win_dir: Path,
    filters: Optional[List[str]],        # None = all available filters
    output_dir: Optional[Path] = None,
    session: str = "",
    min_baseline: float = 0.05,
    r_scale: float = 2.0,
) -> List[ApertureContrastResult]:
    """Compute aperture contrast for all on-disk satellites/shadows in one window pair.

    Iterates over every filter TIF present in the window directory (or only those
    listed in *filters* when provided).  Tracker params are read once from the log
    and reused across filters; disk center is resolved per-filter.
    """
    try:
        from pipeline.modules.satellite_tracker import SatelliteTracker
        from pipeline.modules.derotation import find_disk_center
        from pipeline.modules.image_io import read_tif
    except ImportError as exc:
        raise ImportError(f"Pipeline modules required for auto mode: {exc}") from exc

    log_path = comp_win_dir / "derotation_log.json"
    if not log_path.exists():
        log_path = base_win_dir / "derotation_log.json"
    if not log_path.exists():
        return []

    with open(log_path) as f:
        log = json.load(f)

    center_time = datetime.strptime(log["center_time"], "%Y-%m-%dT%H:%M:%SZ")
    session_log = log.get("session", {})
    sat_log     = log.get("satellite", {})

    pole_pa = float(sat_log.get("pole_pa_deg",    session_log.get("pole_pa_deg",    0.0)))
    np_ang  = float(sat_log.get("np_ang_deg",     0.0))
    flip_ns = bool( sat_log.get("tracker_flip_ns", session_log.get("tracker_flip_ns", False)))
    flip_ew = bool( session_log.get("flip_ew", False))

    tracker = SatelliteTracker(flip_ew=flip_ew, flip_ns=flip_ns)
    ps_calib = session_log.get("plate_scale_calibration")
    if ps_calib:
        tracker.set_plate_scale_calibration(
            float(ps_calib["ps_fit"]), float(ps_calib["cx_offset"])
        )

    # Collect TIFs to process: all in composite dir, optionally filtered
    all_comp_tifs = sorted(comp_win_dir.glob("*_derotated.tif"))
    if filters:
        all_comp_tifs = [t for t in all_comp_tifs
                         if t.stem.replace("_derotated", "") in filters]
    if not all_comp_tifs:
        return []

    win_name  = comp_win_dir.name                          # "window_01"
    win_label = f"{session}_{win_name}" if session else win_name   # "260321_window_01"

    def _compositing_was_applied(img_b, img_c, cx, cy, r) -> bool:
        mask = _annular_mask(img_b.shape, cx, cy, 0.0, 2.5 * r)
        return not np.allclose(img_b[mask], img_c[mask], atol=1e-6)

    results: List[ApertureContrastResult] = []

    for comp_tif in all_comp_tifs:
        filt_name = comp_tif.stem.replace("_derotated", "")
        base_tif  = base_win_dir / comp_tif.name
        if not base_tif.exists():
            continue

        try:
            img_comp = read_tif(comp_tif)
            img_base = read_tif(base_tif)
            _stored  = sat_log.get("disk_centers", {}).get(filt_name)
            if _stored:
                cx = float(_stored["cx"])
                cy = float(_stored["cy"])
                r  = float(_stored["r"])
            else:
                cx, cy, r, *_ = find_disk_center(img_base)
            # Wavelet sharpening applied to both images before contrast measurement
            img_base = _apply_wavelet_display(img_base, cx, cy, r)
            img_comp = _apply_wavelet_display(img_comp, cx, cy, r)
        except Exception:
            continue

        plate_scale = tracker.get_plate_scale(r, center_time)
        print(f"    [{filt_name}] pole_pa={pole_pa:.1f}°  np_ang={np_ang:.3f}°  "
              f"flip_ns={flip_ns}  ps={plate_scale:.4f}\"/px"
              + (f"  cx_offset={ps_calib['cx_offset']:+.2f}px" if ps_calib else ""))

        body_pos = tracker.get_positions(
            [center_time], cx, cy, r, pole_pa_deg=pole_pa, np_ang_deg=np_ang
        )
        try:
            shad_pos = tracker.get_shadow_positions(
                [center_time], cx, cy, r, pole_pa_deg=pole_pa, np_ang_deg=np_ang
            ) or {}
        except Exception:
            shad_pos = {}

        for moon, plist in {**body_pos, **shad_pos}.items():
            p = plist[0] if plist else None
            if p is not None and p.on_disk:
                print(f"      [{moon}] px=({p.x_px:.1f}, {p.y_px:.1f})  "
                      f"dist={p.dist_px:.1f}px")

        for moon_name, pos_list in body_pos.items():
            p = pos_list[0] if pos_list else None
            if p is None or not p.on_disk:
                continue
            app_r = _apparent_radius_px(moon_name, center_time, plate_scale)
            r_ref = max(app_r * r_scale, 3.0)
            if not _compositing_was_applied(img_base, img_comp, p.x_px, p.y_px, r_ref):
                continue
            label  = f"{win_label}_{filt_name}_{moon_name}"
            c_base = measure_aperture_contrast(img_base, p.x_px, p.y_px, r_ref, "satellite")
            c_comp = measure_aperture_contrast(img_comp, p.x_px, p.y_px, r_ref, "satellite")
            if np.isnan(c_base) or c_base < min_baseline:
                print(f"      [{moon_name}] skip: baseline={c_base:.4f} < min_baseline={min_baseline}")
                continue
            res = ApertureContrastResult(
                label=label, obj_type="satellite",
                x_px=p.x_px, y_px=p.y_px, r_px=r_ref,
                contrast_baseline=c_base, contrast_composite=c_comp,
                filter_name=filt_name, session=session, window=win_name,
            )
            results.append(res)
            if output_dir is not None:
                _save_comparison_image(res, img_base, img_comp, output_dir,
                                       disk_cx=cx, disk_cy=cy, disk_r=0.0)

        for shad_name, pos_list in shad_pos.items():
            p = pos_list[0] if pos_list else None
            if p is None or not p.on_disk:
                continue
            moon_name = shad_name.replace("_shadow", "")
            app_r = _apparent_radius_px(moon_name, center_time, plate_scale)
            r_ref = max(app_r * r_scale, 3.0)
            if not _compositing_was_applied(img_base, img_comp, p.x_px, p.y_px, r_ref):
                continue
            label  = f"{win_label}_{filt_name}_{shad_name}"
            c_base = measure_aperture_contrast(img_base, p.x_px, p.y_px, r_ref, "shadow")
            c_comp = measure_aperture_contrast(img_comp, p.x_px, p.y_px, r_ref, "shadow")
            if np.isnan(c_base) or c_base < min_baseline:
                print(f"      [{shad_name}] skip: baseline={c_base:.4f} < min_baseline={min_baseline}")
                continue
            res = ApertureContrastResult(
                label=label, obj_type="shadow",
                x_px=p.x_px, y_px=p.y_px, r_px=r_ref,
                contrast_baseline=c_base, contrast_composite=c_comp,
                filter_name=filt_name, session=session, window=win_name,
            )
            results.append(res)
            if output_dir is not None:
                _save_comparison_image(res, img_base, img_comp, output_dir,
                                       disk_cx=cx, disk_cy=cy, disk_r=0.0)

    return results


def run_from_dirs(
    baseline_dir: Path,
    composite_dir: Path,
    filters: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    session: str = "",
    min_baseline: float = 0.05,
    r_scale: float = 2.0,
) -> List[ApertureContrastResult]:
    """Compute aperture contrast from existing step04 output directories.

    Non-overlapping windows are selected by quality (greedy, same as Step 9).
    filters: list of filter names to measure (None = all available).
    """
    def _resolve(d: Path) -> Path:
        if (d / "step04_derotated").is_dir():
            return d / "step04_derotated"
        return d

    baseline_dir  = _resolve(baseline_dir)
    composite_dir = _resolve(composite_dir)

    all_comp_wins = sorted(composite_dir.glob("window_*"))
    if not all_comp_wins:
        raise FileNotFoundError(f"No window_XX directories in {composite_dir}")

    selected_comp_wins = _deoverlap_win_dirs(all_comp_wins)
    skipped = len(all_comp_wins) - len(selected_comp_wins)
    if skipped:
        print(f"  [overlap] {skipped} overlapping window(s) excluded "
              f"({len(selected_comp_wins)} non-overlapping retained)")

    results: List[ApertureContrastResult] = []
    for comp_win in selected_comp_wins:
        base_win = baseline_dir / comp_win.name
        if not base_win.is_dir():
            print(f"  [skip] {comp_win.name}: no matching baseline window")
            continue
        print(f"  Processing {comp_win.name}…")
        win_results = _measure_window_pair(base_win, comp_win, filters, output_dir,
                                           session=session, min_baseline=min_baseline, r_scale=r_scale)
        if win_results:
            results.extend(win_results)
        else:
            print(f"    → no on-disk satellite/shadow found, skipped")

    return results


def run_from_tif_dir(
    tif_dir: Path,
    filters: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    session: str = "",
    window_frames: int = 3,
    min_baseline: float = 0.05,
    r_scale: float = 2.0,
) -> List[ApertureContrastResult]:
    """Run step3 + step4×2 and compute aperture contrast.

    tif_dir:       Step 3 input folder — the same folder selected in the GUI.
    filters:       Filter names to measure (None = all available).
    output_dir:    Where to save everything (step4 outputs, logs, comparison PNGs).
                   If None, a temporary directory is used and cleaned up afterward.
    window_frames: Number of filter cycles per de-rotation window (default 3 → ~11.25 min).

    Only non-overlapping windows with on-disk satellites/shadows are processed.
    """
    try:
        from pipeline.config import PipelineConfig
        from pipeline.steps import quality_assess, derotate_stack
        from pipeline.modules.image_io import group_by_filter
    except ImportError as exc:
        raise ImportError(f"Pipeline package required for --tif-dir mode: {exc}") from exc

    groups = group_by_filter(tif_dir)
    if not groups:
        raise FileNotFoundError(f"No recognised TIF files in {tif_dir}")
    available_filters = list(groups.keys())
    print(f"[aperture_contrast] Detected filters: {available_filters}")

    _tmp = None
    if output_dir is None:
        _tmp = tempfile.mkdtemp(prefix="aperture_contrast_")
        output_dir = Path(_tmp)
    else:
        print(f"[aperture_contrast] Output → {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_out  = output_dir / "baseline"
    composite_out = output_dir / "composite"
    # PNGs go directly into output_dir (alongside the step4 subdirs)
    img_out = output_dir if _tmp is None else None

    def _make_config(out_base: Path, composite_enabled: bool) -> PipelineConfig:
        cfg = PipelineConfig(
            input_dir=tif_dir,
            output_base_dir=out_base,
            filters=available_filters,
            save_step03=False,
            save_step04=True,
        )
        cfg.quality.window_frames       = window_frames
        cfg.satellite.enabled           = True
        cfg.satellite.composite_enabled = composite_enabled
        return cfg

    # Step 3
    print("\n[aperture_contrast] Step 3: quality assessment & windowing…")
    cfg_run    = _make_config(baseline_out, False)
    results_03 = quality_assess.run(cfg_run)

    all_windows = results_03.get("windows", [])
    if not all_windows:
        raise RuntimeError("Step 3 found no valid time windows — check TIF files.")

    # De-overlap windows (greedy, by quality, same as Step 9)
    selected_windows = _deoverlap_windows(all_windows)
    skipped = len(all_windows) - len(selected_windows)
    if skipped:
        print(f"[aperture_contrast] {skipped} overlapping window(s) excluded "
              f"({len(selected_windows)} non-overlapping retained)")

    # Keep only windows with on-disk satellite/shadow
    sat_windows = _filter_satellite_windows(selected_windows, groups)
    skipped_nosat = len(selected_windows) - len(sat_windows)
    if skipped_nosat:
        print(f"[aperture_contrast] {skipped_nosat} window(s) with no on-disk "
              f"satellite/shadow skipped ({len(sat_windows)} remaining)")
    if not sat_windows:
        print("[aperture_contrast] No windows with on-disk satellites/shadows found.")
        return []

    # Inject filtered windows back for step4
    results_03_filtered = {**results_03, "windows": sat_windows}

    # Step 4 — baseline
    print(f"\n[aperture_contrast] Step 4 (baseline) — {len(sat_windows)} window(s)…")
    cfg_base = _make_config(baseline_out, False)
    derotate_stack.run(cfg_base, results_03_filtered)

    # Step 4 — composite
    print(f"\n[aperture_contrast] Step 4 (composite) — {len(sat_windows)} window(s)…")
    cfg_comp = _make_config(composite_out, True)
    derotate_stack.run(cfg_comp, results_03_filtered)

    print("\n[aperture_contrast] Measuring aperture contrast…")
    results = run_from_dirs(
        baseline_dir=baseline_out,
        composite_dir=composite_out,
        filters=filters,
        output_dir=img_out,
        session=session or (tif_dir.parent.name if tif_dir.name.lower() in ("tifs", "tif") else tif_dir.name),
        min_baseline=min_baseline, r_scale=r_scale,
    )

    if _tmp is not None:
        import shutil
        shutil.rmtree(_tmp, ignore_errors=True)

    return results


# ── Multi-session batch ───────────────────────────────────────────────────────

def run_multi_session(
    tif_dirs: List[Path],
    filters: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    window_frames: int = 3,
    min_baseline: float = 0.05,
    r_scale: float = 2.0,
) -> Dict[str, List[ApertureContrastResult]]:
    """Run aperture contrast for multiple sessions.

    Each tif_dir is processed independently (step3 + step4×2).
    Returns {session_label: [results]} ordered by tif_dir order.
    """
    # When all dirs share the same leaf name (e.g. all "TIFs"), use parent dir name
    # as the session label so outputs don't collide.
    leaf_names = [d.name for d in tif_dirs]
    use_parent_label = len(set(leaf_names)) == 1

    all_results: Dict[str, List[ApertureContrastResult]] = {}
    for tif_dir in tif_dirs:
        label = tif_dir.parent.name if use_parent_label else tif_dir.name
        sess_out = (output_dir / label) if output_dir is not None else None
        print(f"\n{'='*60}")
        print(f"[multi-session] Session: {label}")
        print(f"{'='*60}")
        try:
            results = run_from_tif_dir(
                tif_dir, filters=filters, output_dir=sess_out, session=label,
                window_frames=window_frames, min_baseline=min_baseline, r_scale=r_scale,
            )
        except Exception as exc:
            print(f"  [ERROR] {label}: {exc}")
            results = []
        all_results[label] = results
    return all_results


def print_multi_summary(
    results_by_session: Dict[str, List[ApertureContrastResult]],
) -> None:
    """Print a compact multi-session aperture contrast table.

    Rows: one per (session, window, object).
    Per-filter and overall summary appended at the end.
    """
    all_results = [r for rs in results_by_session.values() for r in rs]
    if not all_results:
        print("No results.")
        return

    valid = [r for r in all_results if not np.isnan(r.delta_pct)]

    # ── Collect all filter names in preferred order ────────────────────────────
    _FILT_ORDER = ["IR", "R", "G", "B", "CH4", "color"]
    seen_filts = []
    for r in all_results:
        if r.filter_name and r.filter_name not in seen_filts:
            seen_filts.append(r.filter_name)
    ordered_filts = [f for f in _FILT_ORDER if f in seen_filts] + \
                    [f for f in seen_filts if f not in _FILT_ORDER]

    # ── Per-session detail table ───────────────────────────────────────────────
    W_SES = 10; W_WIN = 10; W_OBJ = 18; W_FLT = 6
    hdr = (f"{'Session':<{W_SES}}  {'Window':<{W_WIN}}  {'Object':<{W_OBJ}}  "
           f"{'Filter':<{W_FLT}}  {'Type':<9}  {'Baseline':>9}  {'Composite':>9}  {'Δ':>7}")
    sep = "─" * len(hdr)

    print(f"\n{'═'*len(hdr)}")
    print("  Per-window Aperture Contrast")
    print(f"{'═'*len(hdr)}")
    print(hdr)
    print(sep)

    prev_session = None
    for r in all_results:
        if r.session != prev_session:
            if prev_session is not None:
                print(sep)
            prev_session = r.session
        # Derive window and object from dedicated fields (label may have session prefix)
        win = r.window if r.window else r.label
        # Strip "{session}_{window}_{filter}_" prefix from label to get object name
        prefix = f"{r.session}_{r.window}_{r.filter_name}_" if r.window else ""
        obj = r.label[len(prefix):] if prefix and r.label.startswith(prefix) else r.label
        delta  = f"{r.delta_pct:+7.1f}%" if not np.isnan(r.delta_pct) else "     NaN"
        print(
            f"{r.session:<{W_SES}}  {win:<{W_WIN}}  {obj:<{W_OBJ}}  "
            f"{r.filter_name:<{W_FLT}}  {r.obj_type:<9}  "
            f"{r.contrast_baseline:9.4f}  {r.contrast_composite:9.4f}  {delta}"
        )
    print(f"{'═'*len(hdr)}")

    def _fmt(subset):
        if not subset:
            return "           —"
        deltas = [r.delta_pct for r in subset]
        return f"{np.mean(deltas):+7.1f}±{np.std(deltas):.1f}%"

    # ── Per-filter summary ─────────────────────────────────────────────────────
    if ordered_filts:
        print(f"\n{'─'*72}")
        print(f"  Per-filter summary")
        print(f"{'─'*72}")
        hdr2 = (f"{'Filter':<6}  {'N_sat':>5}  {'Satellite Δ':>14}  "
                f"{'N_shd':>5}  {'Shadow Δ':>12}  {'Overall Δ':>12}")
        print(hdr2)
        print("─" * len(hdr2))

        for filt in ordered_filts:
            filt_all  = [r for r in valid if r.filter_name == filt]
            filt_body = [r for r in filt_all if r.obj_type == "satellite"]
            filt_shad = [r for r in filt_all if r.obj_type == "shadow"]
            if not filt_all:
                continue
            print(
                f"{filt:<6}  {len(filt_body):>5}  "
                f"{_fmt(filt_body):>14}  {len(filt_shad):>5}  "
                f"{_fmt(filt_shad):>12}  {_fmt(filt_all):>12}"
            )

    # ── Per-session × per-object summary ─────────────────────────────────────
    def _obj_name(r: ApertureContrastResult) -> str:
        prefix = f"{r.session}_{r.window}_{r.filter_name}_"
        return r.label[len(prefix):] if r.label.startswith(prefix) else r.label

    session_order = list(results_by_session.keys())
    if len(session_order) > 1:
        # Collect (session, object_name) pairs in encounter order
        seen: list = []
        for r in valid:
            key = (r.session, _obj_name(r))
            if key not in seen:
                seen.append(key)

        W_FILT = 13
        filt_cols = ordered_filts if ordered_filts else []
        filt_hdr  = "  ".join(f"{f:>{W_FILT}}" for f in filt_cols)
        hdr3 = f"{'Session':<10}  {'Object':<18}  {'Type':<9}  {filt_hdr}"
        sep3  = "─" * len(hdr3)
        print(f"\n{sep3}")
        print(f"  Per-session / per-object summary  (format: N×mean Δ%)")
        print(sep3)
        print(hdr3)
        print(sep3)

        def _cell(subset):
            if not subset:
                return "—".rjust(W_FILT)
            deltas = [r.delta_pct for r in subset]
            return f"{len(subset)}×{np.mean(deltas):+.1f}%".rjust(W_FILT)

        prev_sess = None
        for sess, obj in seen:
            if prev_sess is not None and sess != prev_sess:
                print(sep3)
            prev_sess = sess
            subset   = [r for r in valid if r.session == sess and _obj_name(r) == obj]
            obj_type = subset[0].obj_type if subset else "—"
            cells = "  ".join(
                _cell([r for r in subset if r.filter_name == f])
                for f in filt_cols
            )
            print(f"{sess:<10}  {obj:<18}  {obj_type:<9}  {cells}")

    # ── Overall ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Overall (N={len(valid)})")
    print(f"{'─'*60}")
    for obj_type in ("satellite", "shadow"):
        subset = [r for r in valid if r.obj_type == obj_type]
        if subset:
            deltas = [r.delta_pct for r in subset]
            print(f"  {obj_type.capitalize():<10} N={len(subset):>3}  "
                  f"mean Δ = {np.mean(deltas):+.1f}%  std = {np.std(deltas):.1f}%")
    if valid:
        all_deltas = [r.delta_pct for r in valid]
        print(f"  {'All':<10} N={len(valid):>3}  "
              f"mean Δ = {np.mean(all_deltas):+.1f}%  std = {np.std(all_deltas):.1f}%")


# ── Batch from CSV ────────────────────────────────────────────────────────────

def run_csv(csv_path: Path) -> List[ApertureContrastResult]:
    results = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = measure_pair(
                baseline_path=Path(row["baseline"]),
                composite_path=Path(row["composite"]),
                x_px=float(row["x_px"]),
                y_px=float(row["y_px"]),
                r_px=float(row["r_px"]),
                obj_type=row["type"],
                label=row.get("label", ""),
            )
            results.append(r)
    return results


def print_summary(results: List[ApertureContrastResult]) -> None:
    if not results:
        print("No results.")
        return

    valid = [r for r in results if not np.isnan(r.delta_pct)]

    print("\n" + "=" * 75)
    print(f"{'Label':<35}  {'Baseline':>9}  {'Composite':>9}  {'Δ':>7}")
    print("-" * 75)
    for r in results:
        delta_str = f"{r.delta_pct:+7.1f}%" if not np.isnan(r.delta_pct) else "     NaN"
        print(
            f"{r.label:<35}  "
            f"{r.contrast_baseline:9.4f}  "
            f"{r.contrast_composite:9.4f}  "
            f"{delta_str}"
        )

    if valid:
        mean_base  = float(np.mean([r.contrast_baseline for r in valid]))
        mean_comp  = float(np.mean([r.contrast_composite for r in valid]))
        mean_delta = float(np.mean([r.delta_pct for r in valid]))
        avg_label  = f"Average (N={len(valid)})"
        print("-" * 75)
        print(
            f"{avg_label:<35}  "
            f"{mean_base:9.4f}  "
            f"{mean_comp:9.4f}  "
            f"{mean_delta:+7.1f}%"
        )
    print("=" * 75)

    # Per-type breakdown (shown only when both types are present)
    types_present = {r.obj_type for r in valid}
    if len(types_present) > 1:
        for obj_type in ("satellite", "shadow"):
            subset = [r for r in valid if r.obj_type == obj_type]
            if not subset:
                continue
            deltas = np.array([r.delta_pct for r in subset])
            print(
                f"\n{obj_type.capitalize()} (N={len(subset)}): "
                f"mean Δ = {deltas.mean():+.1f}%  "
                f"std = {deltas.std():.1f}%"
            )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Aperture contrast measurement for multi-rate de-rotation evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--tif-dir", metavar="DIR", nargs="+",
        help="Step 3 input TIF folder(s) — runs step3+step4×2 per session. "
             "Multiple dirs = multi-session batch.",
    )
    mode.add_argument(
        "--baseline-dir", metavar="DIR",
        help="step04_derotated dir from a baseline run (composite_enabled=False)",
    )
    mode.add_argument(
        "--csv", metavar="FILE",
        help="Batch CSV with columns: baseline,composite,x_px,y_px,r_px,type,label",
    )
    mode.add_argument(
        "--baseline", metavar="FILE",
        help="Baseline (no composite) TIF image",
    )

    p.add_argument("--composite-dir", metavar="DIR",
                   help="step04_derotated dir from a composite run (composite_enabled=True); "
                        "required with --baseline-dir")
    p.add_argument("--filter", metavar="NAME", nargs="*", default=None,
                   help="Filter name(s) to measure, e.g. --filter IR CH4. "
                        "Omit to measure all available filters.")
    p.add_argument("--r-scale", metavar="X", type=float, default=2.0,
                   help="Multiply apparent radius by X to get aperture reference radius r "
                        "(default 2.0). Larger values push the background annulus further "
                        "from the satellite, reducing PSF contamination.")
    p.add_argument("--min-baseline", metavar="C", type=float, default=0.05,
                   help="Exclude measurements where baseline contrast < C "
                        "(default 0.05). Filters out noise-floor cases where "
                        "%% change is statistically meaningless.")
    p.add_argument("--window-frames", metavar="N", type=int, default=3,
                   help="Number of filter cycles per de-rotation window "
                        "(default 3 → 3 × cycle_min ≈ 11.25 min for a 5-filter session).")
    p.add_argument("--output-dir", metavar="DIR", default=None,
                   help="Directory for all outputs: step4 stacks, derotation logs, "
                        "and comparison PNGs. Required to preserve results on disk.")

    p.add_argument("--composite", metavar="FILE",
                   help="Composite (multi-rate stacked) TIF image")
    p.add_argument("--x", type=float, metavar="PX",
                   help="Satellite/shadow center X pixel")
    p.add_argument("--y", type=float, metavar="PX",
                   help="Satellite/shadow center Y pixel")
    p.add_argument("--r", type=float, metavar="PX",
                   help="Reference radius r in pixels: 10%% of the satellite/shadow "
                        "apparent disk diameter (e.g. diameter=184px → r=18.4)")
    p.add_argument("--type", choices=["satellite", "shadow"], default="satellite",
                   help="Object type: 'satellite' (uses ROI max) or 'shadow' (uses ROI min)")
    p.add_argument("--label", default="", help="Label for output table")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else None
    # --filter with nargs="*": None → not given (all), [] → given with no args (all)
    filters: Optional[List[str]] = args.filter if args.filter else None
    window_frames: int = args.window_frames
    min_baseline: float = args.min_baseline
    r_scale: float = args.r_scale

    if args.tif_dir:
        tif_dirs = [Path(d) for d in args.tif_dir]
        if len(tif_dirs) == 1:
            results = run_from_tif_dir(
                tif_dir=tif_dirs[0],
                filters=filters,
                output_dir=out_dir,
                session=tif_dirs[0].name,
                window_frames=window_frames,
                min_baseline=min_baseline, r_scale=r_scale,
            )
            print_summary(results)
        else:
            results_by_session = run_multi_session(
                tif_dirs=tif_dirs,
                filters=filters,
                output_dir=out_dir,
                window_frames=window_frames,
                min_baseline=min_baseline, r_scale=r_scale,
            )
            print_multi_summary(results_by_session)

    elif args.baseline_dir:
        if not args.composite_dir:
            parser.error("--composite-dir is required with --baseline-dir")
        results = run_from_dirs(
            baseline_dir=Path(args.baseline_dir),
            composite_dir=Path(args.composite_dir),
            filters=filters,
            output_dir=out_dir,
        )
        print_summary(results)

    elif args.csv:
        results = run_csv(Path(args.csv))
        print_summary(results)

    else:
        for flag in ("composite", "x", "y", "r"):
            if getattr(args, flag) is None:
                parser.error(f"--{flag} is required when using --baseline")
        results = [measure_pair(
            baseline_path=Path(args.baseline),
            composite_path=Path(args.composite),
            x_px=args.x,
            y_px=args.y,
            r_px=args.r,
            obj_type=args.type,
            label=args.label or Path(args.baseline).stem,
        )]
        print_summary(results)


if __name__ == "__main__":
    main()
