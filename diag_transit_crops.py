#!/usr/bin/env python3
"""
Transit crop diagnostic.

For each TIF in Derotation_Paper_Dataset/260321/TIFs/:
  1. Parse timestamp from filename.
  2. Predict satellite body + shadow positions via SatelliteTracker.
  3. Skip if nothing is on-disk.
  4. Apply wavelet sharpening [200, 200, 200, 0, 0, 0].
  5. Crop 50×50 px around each transit position, scale 2×, draw red crosshair.
  6. Save to ./diag/<stem>_<name>.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# Project root on path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.modules import image_io
from pipeline.modules.derotation import (
    _to_luminance,
    auto_detect_pole_pa,
    find_disk_center,
    pole_pa_from_disk_ellipse,
    query_horizons_np_ang,
)
from pipeline.modules.satellite_tracker import SatelliteTracker
from pipeline.modules.wavelet import sharpen

# ── Config ────────────────────────────────────────────────────────────────────

import argparse as _ap
_parser = _ap.ArgumentParser(description="Transit crop diagnostic")
_parser.add_argument("tif_dir", nargs="?",
                     default="Derotation_Paper_Dataset/260321/TIFs",
                     help="Path to TIF folder (default: 260321/TIFs)")
_args, _ = _parser.parse_known_args()

TIF_DIR  = Path(_args.tif_dir)
DIAG_DIR = Path(__file__).parent / "diag"

HORIZONS_ID     = "599"       # Jupiter
OBSERVER        = "500@399"   # geocentric
WAVELET_AMOUNTS = [200.0, 200.0, 200.0, 0.0, 0.0, 0.0]
CROP_HALF       = 50          # half-side of crop → 100×100
CROP_SCALE      = 2           # upscale factor → 200×200


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_bgr_u8(img_f32: np.ndarray) -> np.ndarray:
    """Convert float32 [0,1] grayscale to BGR uint8 for OpenCV drawing."""
    u8 = np.clip(img_f32 * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR) if u8.ndim == 2 else u8


def _draw_crosshair(
    bgr: np.ndarray,
    x: float,
    y: float,
    size: int = 20,
    color: tuple = (0, 0, 255),
    thickness: int = 1,
) -> None:
    ix, iy = int(round(x)), int(round(y))
    h, w = bgr.shape[:2]
    cv2.line(bgr, (max(0, ix - size), iy), (min(w - 1, ix + size), iy),
             color, thickness, cv2.LINE_AA)
    cv2.line(bgr, (ix, max(0, iy - size)), (ix, min(h - 1, iy + size)),
             color, thickness, cv2.LINE_AA)


def _make_crop(
    bgr_full: np.ndarray,
    x: float,
    y: float,
    half: int = CROP_HALF,
    scale: int = CROP_SCALE,
) -> np.ndarray:
    """
    Crop (2*half × 2*half) centred at (x, y), pad if near edge,
    upscale by *scale*, draw red crosshair at centre.
    """
    h, w = bgr_full.shape[:2]
    ix, iy = int(round(x)), int(round(y))

    x1, x2 = max(0, ix - half), min(w, ix + half)
    y1, y2 = max(0, iy - half), min(h, iy + half)
    crop = bgr_full[y1:y2, x1:x2].copy()

    pl = max(0, half - ix)
    pr = max(0, ix + half - w)
    pt = max(0, half - iy)
    pb = max(0, iy + half - h)
    if pl or pr or pt or pb:
        crop = cv2.copyMakeBorder(crop, pt, pb, pl, pr,
                                  cv2.BORDER_CONSTANT, value=0)

    big = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale),
                     interpolation=cv2.INTER_LINEAR)

    # Crosshair at crop centre (== predicted position)
    _draw_crosshair(big, half * scale, half * scale, size=15)
    return big


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    DIAG_DIR.mkdir(exist_ok=True)

    tifs = sorted(p for p in TIF_DIR.glob("*.tif"))
    if not tifs:
        print(f"No TIF files found in {TIF_DIR}")
        return
    print(f"Found {len(tifs)} TIF files in {TIF_DIR}")

    # ── Try to load session params from an existing derotation_log.json ─────────
    # Priority: derotation_log.json > computation.
    # Looks for step04_derotated/**/derotation_log.json relative to TIF_DIR's
    # parent (session root), then reads pole_pa_deg and tracker_flip_ns from
    # the session block — the same values the pipeline wrote during step 4.
    session_root  = TIF_DIR.parent
    log_candidates = sorted(session_root.rglob("derotation_log.json"))
    pole_pa:   float | None = None
    flip_ns:   bool         = False
    log_source: str         = "computation"

    if log_candidates:
        try:
            import json as _json
            log_data = _json.loads(log_candidates[0].read_text(encoding="utf-8"))
            sess = log_data.get("session", {})
            if "pole_pa_deg" in sess:
                pole_pa    = float(sess["pole_pa_deg"])
                flip_ns    = bool(sess.get("tracker_flip_ns", False))
                log_source = str(log_candidates[0].resolve())
                print(f"  [session] Loaded from derotation_log: {log_source}")
                print(f"    pole_pa_deg   = {pole_pa}°")
                print(f"    tracker_flip_ns = {flip_ns}")
        except Exception as e:
            print(f"  [session] derotation_log read failed ({e}) — falling back to computation")

    # ── Pick preferred filter TIFs (mirrors _FILT_PREF_EXT in derotate_stack.py) ──
    FILT_PREF = ["IR", "R", "G", "B", "CH4"]
    pref_tifs: list[Path] = []
    for filt in FILT_PREF:
        pref_tifs = sorted(p for p in tifs if f"-U-{filt}-" in p.name)
        if pref_tifs:
            print(f"  Session filter for disk/PA scan: {filt} ({len(pref_tifs)} frames)")
            break
    if not pref_tifs:
        pref_tifs = tifs

    # ── Disk center from middle frame (mirrors _detect_session_flip_ns) ──────────
    mid_path = pref_tifs[len(pref_tifs) // 2]
    mid_meta = image_io.parse_filename(mid_path)
    if mid_meta is None:
        print(f"Cannot parse timestamp from {mid_path.name} — aborting")
        return

    raw_mid = image_io.read_tif(mid_path)
    lum_mid = raw_mid if raw_mid.ndim == 2 else raw_mid.mean(axis=2).astype(np.float32)
    cx_ref, cy_ref, r_ref, *_ = find_disk_center(lum_mid)
    if r_ref < 5:
        print("Disk detection failed — aborting")
        return

    t_mid = mid_meta["timestamp"].replace(tzinfo=None)

    # ── Pole PA: from log if available, else _scan_session_pole_pa approach ─────
    if pole_pa is None:
        raw_pas: list[float] = []
        print(f"  [pole_pa] Pre-scanning {len(pref_tifs)} frame(s) for image-space pole PA…")
        for i, p in enumerate(pref_tifs):
            try:
                raw = image_io.read_tif(p)
                lum = raw if raw.ndim == 2 else raw.mean(axis=2).astype(np.float32)
                pa  = auto_detect_pole_pa(frames=[lum], cx=cx_ref, cy=cy_ref,
                                          disk_radius_px=r_ref)
                print(f"    frame {i+1}/{len(pref_tifs)}: pole_pa = {pa:.1f}° [belt_gradient]")
                raw_pas.append(pa)
            except Exception:
                try:
                    pa = pole_pa_from_disk_ellipse(lum)
                    if pa is not None:
                        print(f"    frame {i+1}/{len(pref_tifs)}: pole_pa = {pa:.1f}° [disk_ellipse]")
                        raw_pas.append(pa)
                except Exception:
                    pass
        pole_pa = float(np.median(raw_pas)) if raw_pas else 0.0
        print(f"  [pole_pa] session pole_pa = {pole_pa:.2f}°  (n={len(raw_pas)})")

    # ── NP.ang from bundled table / Horizons (mirrors run() in derotate_stack.py) ──
    np_ang = query_horizons_np_ang(HORIZONS_ID, t_mid, OBSERVER) or 0.0

    # ── Plate scale from SatelliteTracker (mirrors _apply_satellite_composite) ──
    tracker = SatelliteTracker(
        jupiter_horizons_id=HORIZONS_ID,
        observer_code=OBSERVER,
        flip_ew=False,
        flip_ns=flip_ns,
    )
    plate_scale = tracker.get_plate_scale(r_ref, t_mid)

    print(f"  Disk : cx={cx_ref:.1f}  cy={cy_ref:.1f}  r={r_ref:.1f} px")
    print(f"  pole_pa  : {pole_pa:.2f}°  (source: {log_source})")
    print(f"  NP.ang   : {np_ang:.3f}°")
    print(f"  flip_ns  : {flip_ns}")
    print(f"  Plate    : {plate_scale:.5f} arcsec/px")
    print()

    # ── Per-file loop ─────────────────────────────────────────────────────────
    saved = 0
    for tif_path in tifs:
        meta = image_io.parse_filename(tif_path)
        if meta is None:
            print(f"  SKIP {tif_path.name}: cannot parse filename")
            continue

        t = meta["timestamp"].replace(tzinfo=None)

        # Predict positions at this single timestamp
        body_pos = tracker.get_positions(
            [t], cx_ref, cy_ref, r_ref,
            plate_scale_arcsec_per_px=plate_scale,
            pole_pa_deg=pole_pa,
            np_ang_deg=np_ang,
        )
        shad_pos = tracker.get_shadow_positions(
            [t], cx_ref, cy_ref, r_ref,
            plate_scale_arcsec_per_px=plate_scale,
            pole_pa_deg=pole_pa,
            np_ang_deg=np_ang,
        )

        # Collect on-disk transits (body + shadow)
        transits: dict = {}
        for name, poslist in body_pos.items():
            p = poslist[0]
            if p.on_disk:
                transits[name] = p
        for name, poslist in shad_pos.items():
            p = poslist[0]
            if p.on_disk:
                transits[name] = p

        if not transits:
            print(f"  {tif_path.name}: no transit — skip")
            continue

        print(f"  {tif_path.name}: TRANSIT {', '.join(transits)}")

        # Load, sharpen, convert
        raw   = image_io.read_tif(tif_path)
        lum   = (_to_luminance(raw) if raw.ndim == 3
                 else raw.astype(np.float32))
        sharp = sharpen(lum, levels=6, amounts=WAVELET_AMOUNTS)
        bgr   = _to_bgr_u8(sharp)

        # Save one crop per transit body/shadow
        for name, pos in transits.items():
            crop     = _make_crop(bgr, pos.x_px, pos.y_px)
            out_name = f"{tif_path.stem}_{name}.png"
            out_path = DIAG_DIR / out_name
            cv2.imwrite(str(out_path), crop)
            print(f"    → {out_path.name}  "
                  f"(x={pos.x_px:.1f}, y={pos.y_px:.1f}, "
                  f"dist={pos.dist_px:.1f}px from center)")
            saved += 1

    print(f"\nDone. {saved} crop(s) saved to {DIAG_DIR}/")


if __name__ == "__main__":
    main()
