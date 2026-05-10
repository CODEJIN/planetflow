"""
Step 4 – De-rotation stacking.

For each selected time window from Step 3:
  1. Pre-scan all windows to detect session-median image-space pole PA.
  2. Auto-detect camera orientation (flip_ns) from pole PA sign.
  3. Apply spherical de-rotation warp using the correct pole PA.
  4. Sub-pixel translate-align rotated frames via phase correlation.
  5. Combine with quality-weighted mean stack (weights = Step 3 norm_scores).
  6. If config.satellite.enabled: predict Galilean moon/shadow positions via
     JPL Horizons + Skyfield, refine with CV blob detection, and log positions.
  7. If config.satellite.composite_enabled: apply multi-rate compositing (exp9
     method) — overwrite planet TIFs with Europa+shadow composited stacks.

Output (when config.save_step04 is True):
    <output_base>/step04_derotated/
        window_01/
            IR_derotated.tif
            R_derotated.tif
            ...
            derotation_log.json   ← includes satellite positions when enabled
        derotation_summary.txt
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pipeline.config import PipelineConfig
from pipeline.modules import derotation, image_io
from pipeline.modules.derotation import (
    auto_detect_pole_pa,
    find_disk_center,
    query_horizons_np_ang,
)

_FILT_PREF = ["IR", "R", "G", "B", "CH4"]

# Physical mean radii of Galilean moons (km) — used for apparent-size computation
_SATELLITE_RADII_KM: Dict[str, float] = {
    "Io":       1_821.6,
    "Europa":   1_560.8,
    "Ganymede": 2_631.2,
    "Callisto": 2_410.3,
}


# ── Satellite compositing helpers (exp9 method: motion-based Gaussian blend) ──

def _gaussian_mask(shape: Tuple[int, int], cx: float, cy: float, sigma: float) -> np.ndarray:
    H, W = shape
    ys, xs = np.ogrid[:H, :W]
    dist_sq = (xs - cx) ** 2 + (ys - cy) ** 2
    return np.exp(-dist_sq / (2.0 * sigma ** 2)).astype(np.float32)


def _apparent_radius_px(moon_name: str, t_ref, plate_scale: float) -> float:
    """Satellite apparent radius in pixels at t_ref (Skyfield + LTT correction)."""
    from pipeline.modules.satellite_tracker import _load_skyfield_kernels, _MOON_SF_ID
    r_km = _SATELLITE_RADII_KM.get(moon_name, 1_560.8)
    sf = _load_skyfield_kernels()
    if sf is None:
        return 3.0
    ts, eph, jup_moons = sf
    t_sf = ts.utc(t_ref.year, t_ref.month, t_ref.day,
                  t_ref.hour, t_ref.minute, t_ref.second)
    earth_km = eph["earth"].at(t_sf).position.km
    jup_km_t = eph["jupiter barycenter"].at(t_sf).position.km
    d_EJ_km  = float(np.linalg.norm(jup_km_t - earth_km))
    lt_days  = d_EJ_km / (299_792.458 * 86_400.0)
    t_emit   = ts.tt_jd(float(t_sf.tt) - lt_days)
    sf_id    = _MOON_SF_ID.get(moon_name, moon_name.lower())
    moon_km  = jup_moons[sf_id].at(t_emit).position.km
    d_earth_sat = float(np.linalg.norm(moon_km - earth_km))
    return r_km / d_earth_sat * 206_265.0 / plate_scale


def _compute_sigma_from_motion(
    label: str,
    positions: List,
    ref_pos,
    apparent_r_px: float,
    coverage_scale: float,
) -> float:
    """Motion-based Gaussian blend sigma.

    sigma = max(max_motion_px, apparent_radius_px) × coverage_scale

    ref_pos: canonical SatellitePos at window center time (same for all filters).
    coverage_scale=2.5 → α≈0.92 at the farthest streak endpoint (exp9 validated).
    """
    max_motion = 0.0
    if ref_pos is not None:
        for pos in positions:
            if pos is None:
                continue
            d = float(np.hypot(pos.x_px - ref_pos.x_px, pos.y_px - ref_pos.y_px))
            max_motion = max(max_motion, d)
    effective = max(max_motion, apparent_r_px)
    sigma = effective * coverage_scale
    alpha_ep = float(np.exp(-max_motion ** 2 / (2 * sigma ** 2))) if sigma > 0 else 0.0
    print(
        f"      [σ/{label}] apparent_r={apparent_r_px:.2f}px  "
        f"max_motion={max_motion:.2f}px  σ={sigma:.2f}px  α@ep={alpha_ep:.3f}"
    )
    return sigma


def _satellite_translate_stack(
    rows: List[dict], positions: List, ref_pos,
    keep_color: bool = False,
) -> Optional[np.ndarray]:
    """Stack frames by pure translation to align satellite at ref_pos.

    ref_pos: canonical SatellitePos at window center time (same for all filters),
             so all filter stacks place the satellite at the same pixel coordinate.
    No planet warp — only the satellite/shadow region is reliably sharp.
    The planet background is smeared, but it is masked out by the Gaussian blend.

    keep_color: if True, return an (H, W, 3) stack preserving color channels
                (used for color-camera-mode TIFs).
    """
    from pipeline.modules.derotation import apply_shift, quality_weighted_stack
    if ref_pos is None:
        return None
    imgs: List[np.ndarray] = []
    weights: List[float] = []
    for i, row in enumerate(rows):
        pos = positions[i]
        if pos is None:
            continue
        raw = image_io.read_tif(row["path"])
        img = raw.astype(np.float32) / 65535.0 if raw.dtype == np.uint16 else raw.astype(np.float32)
        if img.ndim == 3 and not keep_color:
            img = img.mean(axis=2)
        elif img.ndim == 2 and keep_color:
            img = np.stack([img, img, img], axis=2)
        imgs.append(apply_shift(img, ref_pos.x_px - pos.x_px, ref_pos.y_px - pos.y_px))
        weights.append(float(row["norm_score"]))
    if not imgs:
        return None
    return quality_weighted_stack(imgs, weights)


def _blend_satellite_composite(
    planet: np.ndarray,
    europa_stack: Optional[np.ndarray],
    shadow_stack: Optional[np.ndarray],
    europa_ref,
    shadow_ref,
    europa_sigma: float,
    shadow_sigma: float,
) -> np.ndarray:
    """Blend satellite-derotated patches into planet stack via Gaussian masks.

    Works for both grayscale (H, W) and color (H, W, 3) planet arrays.
    The alpha mask is always computed in 2D and broadcast over channels when needed.
    """
    result = planet.copy()
    is_color = planet.ndim == 3
    shape2d = planet.shape[:2]
    if europa_stack is not None and europa_ref is not None and europa_ref.on_disk:
        alpha = _gaussian_mask(shape2d, europa_ref.x_px, europa_ref.y_px, europa_sigma)
        if is_color:
            alpha = alpha[:, :, np.newaxis]
        result = (1.0 - alpha) * result + alpha * europa_stack
    if shadow_stack is not None and shadow_ref is not None and shadow_ref.on_disk:
        alpha = _gaussian_mask(shape2d, shadow_ref.x_px, shadow_ref.y_px, shadow_sigma)
        if is_color:
            alpha = alpha[:, :, np.newaxis]
        result = (1.0 - alpha) * result + alpha * shadow_stack
    return np.clip(result, 0.0, 1.0)


# ── NOT USED: Approach B — planet warp + satellite translation ─────────────────
#
# Experiment 10 tested applying the same planet de-rotation warp to each
# satellite-stack frame before translating to align the satellite.  The idea
# was that the background texture in the satellite stack would then match the
# planet stack, reducing the mismatch visible at the Gaussian blend boundary.
#
# Results (2026-05-05 Jupiter, Window 3):
#   IR  filter: max pixel difference vs pure-translation (exp9) = 0.12%
#   CH4 filter: max pixel difference vs pure-translation (exp9) = 0.67%
#   Visual: indistinguishable at any filter or zoom level.
#
# Root cause: The warp corrects only ~2.8 px per 4-minute interval at the
# satellite position.  This is far smaller than the stacking-induced background
# smear (which spans the full motion range, e.g. ~6 px for IR).  The smear is
# inherent to stacking N frames with different planet-background offsets; no
# per-frame warp correction can eliminate it without a fundamentally different
# compositing strategy (e.g., background subtraction or inpainting).
#
# Scalability: at 2× pixel scale (C14 + Barlow), both the warp correction and
# the motion range scale proportionally, so the improvement ratio stays ~24%
# and the absolute difference stays below 1.3% — still visually negligible.
#
# When to re-evaluate:
#   - If a background-subtraction or inpainting strategy is adopted, making
#     precise per-frame background texture alignment meaningful.
#   - If spectral analysis (not visual imaging) requires sub-pixel accuracy of
#     the planet background at the blend boundary.
#
# def _warp_displacement_at(x_sat, y_sat, dt_sec, cx, cy, disk_r,
#                            period_hours, warp_scale, pole_pa_deg, polar_eq_ratio):
#     """Return (dx, dy) displacement that planet warp applies at satellite pos."""
#     period_sec = period_hours * 3600.0
#     delta_lambda_rad = (dt_sec / period_sec) * 2.0 * np.pi
#     warp_radius = disk_r * 1.05
#     pole_pa_rad = np.radians(pole_pa_deg)
#     cos_pa = float(np.cos(pole_pa_rad))
#     sin_pa = float(np.sin(pole_pa_rad))
#     rx = x_sat - cx; ry = y_sat - cy
#     rx_eq  = rx * cos_pa + ry * sin_pa
#     ry_pol = -rx * sin_pa + ry * cos_pa
#     polar_scale_sq = (1.0 / max(polar_eq_ratio, 1e-3)) ** 2
#     depth_sq = warp_radius**2 - rx_eq**2 - polar_scale_sq * ry_pol**2
#     depth = float(np.sqrt(max(0.0, depth_sq)))
#     drift = warp_scale * delta_lambda_rad * depth
#     return drift * cos_pa, drift * sin_pa
#
# def _satellite_warp_translate_stack(rows, positions, ref_idx, t_ref,
#                                      disk_cx, disk_cy, disk_sr,
#                                      period_hours, warp_scale,
#                                      pole_pa_deg, polar_eq_ratio):
#     """Approach B: planet warp + additional translation to align satellite.
#
#     Step 1: apply planet de-rotation warp (same as the planet stack).
#     Step 2: compute where satellite lands after warp (analytical displacement).
#     Step 3: translate the residual difference to align satellite at ref pos.
#
#     Max pixel improvement vs pure translation (exp9): 0.12% (IR), 0.67% (CH4).
#     Visually indistinguishable — see NOT USED block above for full analysis.
#     """
#     from pipeline.modules.derotation import (
#         apply_shift, quality_weighted_stack, spherical_derotation_warp,
#     )
#     ref_pos = positions[ref_idx]
#     if ref_pos is None:
#         return None
#     imgs, weights = [], []
#     for i, row in enumerate(rows):
#         pos = positions[i]
#         if pos is None:
#             continue
#         raw = image_io.read_tif(row["path"])
#         img = raw.astype(np.float32) / 65535.0 if raw.dtype == np.uint16 else raw.astype(np.float32)
#         if img.ndim == 3:
#             img = img.mean(axis=2)
#         row_t = row["timestamp"]
#         row_t = row_t.replace(tzinfo=None) if row_t.tzinfo else row_t
#         dt_sec = (row_t - t_ref).total_seconds()
#         warped = spherical_derotation_warp(
#             img, dt_sec, disk_cx, disk_cy, disk_sr,
#             period_hours=period_hours,
#             scale=warp_scale,
#             flip_direction=False,
#             pole_pa_deg=pole_pa_deg,
#             polar_equatorial_ratio=polar_eq_ratio,
#         )
#         wdx, wdy = _warp_displacement_at(
#             pos.x_px, pos.y_px, dt_sec,
#             disk_cx, disk_cy, disk_sr,
#             period_hours, warp_scale, pole_pa_deg, polar_eq_ratio,
#         )
#         sat_x_after_warp = pos.x_px + wdx
#         sat_y_after_warp = pos.y_px + wdy
#         dx = ref_pos.x_px - sat_x_after_warp
#         dy = ref_pos.y_px - sat_y_after_warp
#         imgs.append(apply_shift(warped, dx, dy))
#         weights.append(float(row["norm_score"]))
#     if not imgs:
#         return None
#     return quality_weighted_stack(imgs, weights)
# ─────────────────────────────────────────────────────────────────────────────


def _apply_satellite_composite(
    window: dict,
    filter_results: dict,
    config: "PipelineConfig",
    tracker,
    pole_pa_deg: float,
    np_ang_deg: float,
) -> None:
    """Apply multi-rate satellite compositing to planet-derotated TIFs.

    Each filter's satellite reference position is computed in that filter's own
    disk coordinate system (using that filter's detected disk_cx/cy/sr).  This
    ensures the satellite lands at the same disk-relative position in every filter
    TIF.  When step06's align_channels() shifts non-reference channels to match
    the reference disk position, the satellite shifts by the same amount and
    remains co-located across all channels in the final composite.
    """
    t_center = window["center_time"]
    t_center_naive = t_center.replace(tzinfo=None) if t_center.tzinfo else t_center
    coverage_scale = config.satellite.composite_coverage_scale

    # ── Reference filter: used only for plate_scale / apparent radius ──────────
    ref_filt = next(
        (f for f in _FILT_PREF
         if filter_results.get(f, (None,))[0] is not None
         and filter_results[f][0].exists()),
        None,
    )
    if ref_filt is None:
        return
    ref_tif = image_io.read_tif(filter_results[ref_filt][0])
    ref_lum = ref_tif.astype(np.float32) / 65535.0 if ref_tif.dtype == np.uint16 else ref_tif.astype(np.float32)
    if ref_lum.ndim == 3:
        ref_lum = ref_lum.mean(axis=2)
    _, _, ref_sr, _, _ = derotation.find_disk_center(ref_lum)

    plate_scale = tracker.get_plate_scale(ref_sr, t_center_naive)
    app_r = _apparent_radius_px("Europa", t_center_naive, plate_scale)

    # ── Per-filter satellite composite ─────────────────────────────────────────
    # Each filter uses its OWN disk center for all tracker queries.
    # This ensures the satellite lands at the same disk-relative position in every
    # filter TIF, so step06's disk alignment shift does not create cross-filter
    # satellite position differences in the final composite.
    for filt, (out_path, _) in filter_results.items():
        if out_path is None or not out_path.exists():
            continue
        rows = window.get("per_filter", {}).get(filt, {}).get("included", [])
        if not rows:
            continue

        print(f"    [{filt}] satellite composite…")

        # Load this filter's de-rotated TIF and detect its own disk center
        planet_raw = image_io.read_tif(out_path)
        planet = (planet_raw.astype(np.float32) / 65535.0 if planet_raw.dtype == np.uint16
                  else planet_raw.astype(np.float32))
        is_color = planet.ndim == 3
        planet_lum = planet.mean(axis=2) if is_color else planet
        disk_cx, disk_cy, disk_sr, _, _ = derotation.find_disk_center(planet_lum)

        # Canonical satellite ref at t_center in THIS filter's coordinate system
        canonical_body = tracker.get_positions(
            [t_center_naive], disk_cx, disk_cy, disk_sr,
            pole_pa_deg=pole_pa_deg, np_ang_deg=np_ang_deg,
        )
        canonical_shad = tracker.get_shadow_positions(
            [t_center_naive], disk_cx, disk_cy, disk_sr,
            pole_pa_deg=pole_pa_deg, np_ang_deg=np_ang_deg,
            moon_horizons_positions=canonical_body,
        )
        europa_ref = canonical_body.get("Europa", [None])[0]
        shadow_ref = canonical_shad.get("Europa_shadow", [None])[0]

        time_sorted = sorted(rows, key=lambda r: r["timestamp"])
        t_list = [r["timestamp"] for r in time_sorted]

        body_pos = tracker.get_positions(
            t_list, disk_cx, disk_cy, disk_sr,
            pole_pa_deg=pole_pa_deg, np_ang_deg=np_ang_deg,
        )
        shad_pos = tracker.get_shadow_positions(
            t_list, disk_cx, disk_cy, disk_sr,
            pole_pa_deg=pole_pa_deg, np_ang_deg=np_ang_deg,
            moon_horizons_positions=body_pos,
        )
        europa_positions = body_pos.get("Europa", [None] * len(time_sorted))
        shadow_positions = shad_pos.get("Europa_shadow", [None] * len(time_sorted))

        europa_sigma = _compute_sigma_from_motion("Europa", europa_positions, europa_ref, app_r, coverage_scale)
        shadow_sigma = _compute_sigma_from_motion("Europa_shadow", shadow_positions, shadow_ref, app_r, coverage_scale)

        europa_stack = _satellite_translate_stack(time_sorted, europa_positions, europa_ref, keep_color=is_color)
        shadow_stack = _satellite_translate_stack(time_sorted, shadow_positions, shadow_ref, keep_color=is_color)

        planet_for_blend = planet if is_color else planet_lum
        comp = _blend_satellite_composite(
            planet_for_blend, europa_stack, shadow_stack,
            europa_ref, shadow_ref,
            europa_sigma, shadow_sigma,
        )
        image_io.write_tif_16bit(comp, out_path)
        print(f"      → {out_path.name}  (σ_e={europa_sigma:.1f}px σ_s={shadow_sigma:.1f}px)")


def _scan_session_pole_pa(
    windows: List[dict],
    config: PipelineConfig,
) -> Optional[float]:
    """Pre-scan all windows and return the session-median image-space pole PA.

    Uses the highest-priority filter available per window.  Outliers are kept
    because the camera orientation is fixed within a session.
    """
    raw_pas: List[float] = []
    print("  [pole_pa] Pre-scanning windows for image-space pole PA…")
    for win in windows:
        filt = next(
            (f for f in _FILT_PREF
             if f in win.get("per_filter", {}) and win["per_filter"][f].get("included")),
            None,
        )
        if filt is None:
            continue
        rows = win["per_filter"][filt]["included"]
        t_center = win["center_time"]
        try:
            frames, dts = [], []
            for row in rows:
                raw = image_io.read_tif(row["path"])
                lum = raw if raw.ndim == 2 else raw.mean(axis=2).astype(np.float32)
                frames.append(lum)
                dts.append((row["timestamp"] - t_center).total_seconds())
            cx, cy, semi_a, semi_b, _ = find_disk_center(frames[0])
            polar_eq = float(np.clip(semi_b / max(semi_a, 1.0), 0.85, 1.0))
            pa = auto_detect_pole_pa(
                frames=frames,
                dt_sec_list=dts,
                cx=cx, cy=cy,
                disk_radius_px=semi_a,
                period_hours=config.derotation.rotation_period_hours,
                warp_scale=config.derotation.warp_scale,
                polar_equatorial_ratio=polar_eq,
            )
            print(f"    window {len(raw_pas)+1}: raw pole_pa = {pa:.1f}° via {filt}")
            raw_pas.append(pa)
        except Exception as exc:
            warnings.warn(f"  [pole_pa] window scan failed: {exc}")

    if not raw_pas:
        return None

    session_pa = float(np.median(raw_pas))
    kept = [f"{p:.1f}" for p in raw_pas]
    print(
        f"  [pole_pa] session pole_pa = {session_pa:.1f}° "
        f"(raw: {kept}, kept: {kept})"
    )
    return session_pa


def _precompute_cv_offsets(
    window: dict,
    tracker,
    session_pole_pa: float,
    np_ang: float,
    cv_search_radius_px: float,
    time_offset_sec: float = 0.0,
) -> Optional[Dict[str, Tuple[float, float]]]:
    """Compute per-window CV position offsets from the best available filter."""
    from pipeline.modules.satellite_tracker import (
        refine_positions_with_cv,
        average_body_shadow_offsets,
    )

    filt = next(
        (f for f in _FILT_PREF
         if f in window.get("per_filter", {}) and window["per_filter"][f].get("included")),
        None,
    )
    if filt is None:
        return None

    rows = window["per_filter"][filt]["included"]
    t_center = window["center_time"]
    sorted_rows = sorted(rows, key=lambda r: abs((r["timestamp"] - t_center).total_seconds()))
    try:
        ref_raw = image_io.read_tif(sorted_rows[0]["path"])
        ref_lum = ref_raw if ref_raw.ndim == 2 else ref_raw.mean(axis=2).astype(np.float32)
        cx, cy, sr, *_ = find_disk_center(ref_lum)
        t_list = [r["timestamp"] for r in rows]

        sat_pos = tracker.get_positions(
            t_list, cx, cy, sr,
            pole_pa_deg=session_pole_pa, np_ang_deg=np_ang,
        )
        shd_pos = tracker.get_shadow_positions(
            t_list, cx, cy, sr,
            pole_pa_deg=session_pole_pa, np_ang_deg=np_ang,
            time_offset_sec=time_offset_sec,
        )
        all_pos = {**sat_pos, **shd_pos}

        if not tracker.any_on_disk(all_pos):
            return None

        _, offsets = refine_positions_with_cv(
            ref_lum, cx, cy, sr, all_pos,
            search_radius_px=cv_search_radius_px,
        )
        offsets = average_body_shadow_offsets(offsets)
        print(
            f"  [CV pre-compute/{filt}] offsets: "
            + ", ".join(
                f"{n}: ({dx:+.1f},{dy:+.1f})px"
                for n, (dx, dy) in offsets.items()
                if dx != 0.0 or dy != 0.0
            )
        )
        return offsets
    except Exception as exc:
        warnings.warn(f"  [step04] CV pre-compute failed: {exc}")
        return None


def run(
    config: PipelineConfig,
    results_03: dict,
    progress_callback=None,
    cancel_event=None,
) -> Dict[str, List[Dict]]:
    """Run Step 4 de-rotation stacking.

    Args:
        config:      Pipeline configuration.
        results_03:  Output of step03_quality_assess.run().

    Returns:
        {"windows": [{window_index, center_time, outputs, log}, ...]}
    """
    windows: List[dict] = results_03.get("windows", [])
    if not windows:
        print("  [WARNING] No time windows from Step 3 — de-rotation skipped.")
        return {"windows": []}

    print(f"  Processing {len(windows)} window(s) × {len(config.filters)} filter(s)…")
    print(f"  Period: {config.derotation.rotation_period_hours}h  "
          f"|  warp_scale: {config.derotation.warp_scale}  "
          f"|  sub-pixel alignment: enabled")

    # ── Session-level pole PA ──────────────────────────────────────────────────
    session_pole_pa = _scan_session_pole_pa(windows, config)
    if session_pole_pa is None:
        session_pole_pa = 0.0
        print("  [WARNING] pole_pa scan failed — using 0.0°")

    # ── Auto-detect flip_ns from pole_pa sign ─────────────────────────────────
    sat_cfg = config.satellite
    if sat_cfg.flip_ns is not None:
        flip_ns = sat_cfg.flip_ns
        print(f"  [flip_ns] manual override: {flip_ns}")
    else:
        flip_ns = session_pole_pa < 0.0
        orientation = "South-up" if flip_ns else "North-up"
        print(f"  [flip_ns] auto-detected: pole_pa={session_pole_pa:.1f}° → "
              f"{orientation} (flip_ns={flip_ns})")

    # ── SatelliteTracker ───────────────────────────────────────────────────────
    tracker = None
    if sat_cfg.enabled:
        from pipeline.modules.satellite_tracker import SatelliteTracker
        tracker = SatelliteTracker(
            jupiter_horizons_id=config.derotation.horizons_id,
            observer_code=config.derotation.observer_code,
            flip_ew=sat_cfg.flip_ew,
            flip_ns=flip_ns,
        )
        print(f"  [satellite] tracker enabled  "
              f"(flip_ns={flip_ns}, flip_ew={sat_cfg.flip_ew})")

    # ── Output directory ───────────────────────────────────────────────────────
    out_base: Optional[Path] = None
    if config.save_step04:
        out_base = config.step_dir(4, "derotated")
        out_base.mkdir(parents=True, exist_ok=True)
        print(f"  Output → {out_base}")
    else:
        print("  save_step04=False: results not written to disk")

    # ── Process each window ────────────────────────────────────────────────────
    all_results: List[dict] = []
    summary_lines: List[str] = ["=== Step 4 De-rotation Summary ===\n"]

    n_windows = len(windows)
    for win_idx, window in enumerate(windows, start=1):
        if cancel_event is not None and cancel_event.is_set():
            print("  [CANCELLED] Stopping Step 4.", flush=True)
            break
        if progress_callback is not None:
            progress_callback(win_idx - 1, n_windows)

        t_center = window["center_time"]
        t_center_str = t_center.strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"\n  Window {win_idx}  [{t_center_str}]  "
              f"quality={window['window_quality']:.4f}  "
              f"rotation={window['rotation_degrees']:.1f}°")

        # ── NP.ang from Horizons (celestial North Pole angle) ─────────────────
        np_ang = query_horizons_np_ang(
            horizons_id=config.derotation.horizons_id,
            t_utc=t_center,
            observer_code=config.derotation.observer_code,
        )
        np_ang_val = np_ang if np_ang is not None else 0.0
        if np_ang is None:
            print("    [WARNING] NP.ang not available → using 0.0°")
        else:
            print(f"  [NP.ang = {np_ang_val:.3f}° (celestial)]")

        # pole_pa for the WARP: image-space angle from auto_detect_pole_pa()
        # pole_pa for the TRACKER: pole_pa + NP.ang = camera rotation θ_cam
        pole_pa_for_warp = session_pole_pa
        print(f"  [pole_pa = {pole_pa_for_warp:.1f}° (image-space, for warp)]")

        # Create per-window output directory
        win_out_dir: Optional[Path] = None
        if out_base is not None:
            win_out_dir = out_base / f"window_{win_idx:02d}"
            win_out_dir.mkdir(parents=True, exist_ok=True)

        # ── Satellite position prediction ──────────────────────────────────────
        sat_log: Dict = {}
        cv_offsets: Dict[str, Tuple[float, float]] = {}
        if tracker is not None:
            cv_offsets = _precompute_cv_offsets(
                window, tracker,
                session_pole_pa=pole_pa_for_warp,
                np_ang=np_ang_val,
                cv_search_radius_px=sat_cfg.cv_search_radius_px,
                time_offset_sec=sat_cfg.time_offset_sec,
            ) or {}
            sat_log = {
                "np_ang_deg":  np_ang_val,
                "pole_pa_deg": pole_pa_for_warp,
                "flip_ns":     flip_ns,
                "cv_offsets":  {k: list(v) for k, v in cv_offsets.items()},
            }

        # ── De-rotate all filters ──────────────────────────────────────────────
        filter_results = derotation.derotate_window(
            window=window,
            required_filters=(
                list(window["per_filter"].keys())
                if config.camera_mode == "color"
                else config.filters
            ),
            period_hours=config.derotation.rotation_period_hours,
            warp_scale=config.derotation.warp_scale,
            align=True,
            normalize_brightness=config.derotation.normalize_brightness,
            min_quality_threshold=config.derotation.min_quality_threshold,
            pole_pa_deg=pole_pa_for_warp,
            color_mode=(config.camera_mode == "color"),
            out_dir=win_out_dir,
        )

        # ── Satellite compositing (exp9 method) ───────────────────────────────
        if sat_cfg.composite_enabled and tracker is not None:
            print(f"  [satellite composite] Window {win_idx}…")
            _apply_satellite_composite(
                window=window,
                filter_results=filter_results,
                config=config,
                tracker=tracker,
                pole_pa_deg=pole_pa_for_warp,
                np_ang_deg=np_ang_val,
            )

        # ── Build log and save JSON ────────────────────────────────────────────
        log_dict = derotation.derotation_log_to_json(win_idx, window, filter_results)
        if sat_log:
            log_dict["satellite"] = sat_log
        if win_out_dir is not None:
            json_path = win_out_dir / "derotation_log.json"
            with open(json_path, "w") as f:
                json.dump(log_dict, f, indent=2, default=str)
            print(f"    → {json_path.name}")

        # ── Summary ───────────────────────────────────────────────────────────
        summary_lines.append(
            f"Window {win_idx}  {t_center_str}  "
            f"quality={window['window_quality']:.4f}  "
            f"rotation_span={window['rotation_degrees']:.1f}°"
        )
        summary_filters = (
            list(filter_results.keys()) if config.camera_mode == "color"
            else config.filters
        )
        for filt in summary_filters:
            if filt in filter_results:
                out_path, flog = filter_results[filt]
                n = flog.get("n_stacked", 0)
                snr = round(float(n) ** 0.5, 2)
                fname = out_path.name if out_path else "—"
                summary_lines.append(
                    f"  {filt:>4}: {fname}  ({n} frames, SNR×{snr:.2f})"
                )
            else:
                summary_lines.append(f"  {filt:>4}: not available")
        summary_lines.append("")

        outputs = {filt: res[0] for filt, res in filter_results.items()}
        logs    = {filt: res[1] for filt, res in filter_results.items()}
        all_results.append({
            "window_index": win_idx,
            "center_time":  t_center_str,
            "outputs":      outputs,
            "log":          logs,
            "satellite":    sat_log,
        })

    if progress_callback is not None:
        progress_callback(n_windows, n_windows)

    # ── Save summary ───────────────────────────────────────────────────────────
    summary_text = "\n".join(summary_lines)
    print()
    print(summary_text)
    if out_base is not None:
        txt_path = out_base / "derotation_summary.txt"
        with open(txt_path, "w") as f:
            f.write(summary_text)
        print(f"  → {txt_path}")

    return {"windows": all_results}
