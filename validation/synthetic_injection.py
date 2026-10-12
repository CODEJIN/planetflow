"""
Synthetic injection validation of multi-rate satellite/shadow compositing.

Injects a synthetic satellite body (bright disk) or shadow (dark disk) with a
known trajectory into REAL raw Jupiter frames, then runs the production
Step 4 code path on them and compares the result against a known ground truth:

    P  planet-only stack   derotate_filter() on the ORIGINAL frames
    S  single-rate stack   derotate_filter() on the INJECTED frames
    M  multi-rate stack    S + production compositing (_satellite_translate_stack
                           + _blend_poisson, capsule mask) using the TRUE positions
    T  ground truth        P with the object inserted ONCE at its reference
                           position (what a perfectly smear-free stack would be)

All metrics are computed on object-only residual images (X - P), with no
wavelet sharpening anywhere, so the measured gain cannot come from a
sharpening interaction.

    flux_recovery   sum(R_X) / sum(R_T) inside the aperture  (1.0 = perfect)
    peak_recovery   extremum(R_X) / extremum(R_T)            (1.0 = perfect)
    nrmse           ||R_X - R_T|| / ||R_T|| inside the aperture (0.0 = perfect)
    elongation      major/minor axis ratio of |R_X| (1.0 = round, >1 smeared)
    centroid_err_px distance of the |R_X|-weighted centroid from the true position

A second sweep rotates the positions handed to the compositor about the disk
centre by delta degrees (the injection itself is unchanged), quantifying how
the residual angular placement bias reported in the paper degrades the
compositing result.

Usage:
    python validation/synthetic_injection.py \
        --tif-dir D:/Astro/.../260417/TIFs --out-dir D:/Astro/synthetic_out \
        --filters IR CH4 --n-windows 4
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import shutil
import sys
import zlib
from dataclasses import replace as _dc_replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import PipelineConfig, QualityConfig  # noqa: E402
from pipeline.modules import derotation, image_io  # noqa: E402
from pipeline.modules.satellite_tracker import SatellitePos  # noqa: E402
from pipeline.steps import derotate_stack, quality_assess  # noqa: E402
from pipeline.steps.satellite_composite import (  # noqa: E402
    _blend_poisson,
    _satellite_translate_stack,
)

_COVERAGE_SCALE = 1.5   # production capsule sigma_perp = app_r * 1.5
_PSF_SIGMA_PX = 1.2     # Gaussian seeing blur applied to the injected object


# ── Object model ──────────────────────────────────────────────────────────────

def _object_profile(shape: Tuple[int, int], x: float, y: float, r: float) -> np.ndarray:
    """Anti-aliased uniform disk of radius r at (x, y), blurred by the PSF. Peak ~1."""
    h, w = shape
    ss = 4  # supersampling for sub-pixel accuracy of the disk edge
    yy, xx = np.mgrid[0:h * ss, 0:w * ss].astype(np.float32)
    xx = (xx + 0.5) / ss - 0.5
    yy = (yy + 0.5) / ss - 0.5
    disk = ((xx - x) ** 2 + (yy - y) ** 2 <= r * r).astype(np.float32)
    disk = cv2.resize(disk, (w, h), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(disk, (0, 0), _PSF_SIGMA_PX)


def _inject(img: np.ndarray, x: float, y: float, r: float, obj_type: str,
            amplitude: float) -> np.ndarray:
    """Insert the object into a 2-D [0,1] image.

    satellite: occults the planet and shows its own surface, brightness
               (1 + amplitude) x the local disk level.
    shadow:    multiplicative umbra of depth `amplitude`.
    """
    m = _object_profile(img.shape, x, y, r)
    if obj_type == "satellite":
        ix, iy = int(round(x)), int(round(y))
        r_bg = int(max(4, 3 * r))
        patch = img[max(0, iy - r_bg): iy + r_bg + 1, max(0, ix - r_bg): ix + r_bg + 1]
        level = float(np.median(patch)) * (1.0 + amplitude)
        out = img * (1.0 - m) + level * m
    else:
        out = img * (1.0 - amplitude * m)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ── Metrics ───────────────────────────────────────────────────────────────────

def _metrics(res: np.ndarray, res_truth: np.ndarray, x: float, y: float,
             r_ap: float) -> Dict[str, float]:
    h, w = res.shape
    yy, xx = np.mgrid[0:h, 0:w]
    ap = (xx - x) ** 2 + (yy - y) ** 2 <= r_ap * r_ap
    a, t = res[ap].astype(np.float64), res_truth[ap].astype(np.float64)

    sum_t = float(t.sum())
    ext_t = float(t.min()) if sum_t < 0 else float(t.max())
    ext_a = float(a.min()) if sum_t < 0 else float(a.max())
    flux = float(a.sum()) / sum_t if abs(sum_t) > 1e-12 else float("nan")
    peak = ext_a / ext_t if abs(ext_t) > 1e-12 else float("nan")
    nrmse = float(np.linalg.norm(a - t) / max(np.linalg.norm(t), 1e-12))

    # Shape: second moments of the object-signed residual (clipped at 0 so
    # opposite-sign ringing does not cancel the moments).
    sgn = -1.0 if sum_t < 0 else 1.0
    wgt = np.clip(sgn * res, 0.0, None) * ap
    wsum = float(wgt.sum())
    if wsum > 1e-12:
        mx = float((wgt * xx).sum() / wsum)
        my = float((wgt * yy).sum() / wsum)
        cxx = float((wgt * (xx - mx) ** 2).sum() / wsum)
        cyy = float((wgt * (yy - my) ** 2).sum() / wsum)
        cxy = float((wgt * (xx - mx) * (yy - my)).sum() / wsum)
        ev = np.linalg.eigvalsh(np.array([[cxx, cxy], [cxy, cyy]]))
        elong = float(math.sqrt(max(ev[1], 1e-12) / max(ev[0], 1e-12)))
        cerr = float(math.hypot(mx - x, my - y))
    else:
        elong, cerr = float("nan"), float("nan")
    return {"flux_recovery": flux, "peak_recovery": peak, "nrmse": nrmse,
            "elongation": elong, "centroid_err_px": cerr}


def _artifact_rms(m: np.ndarray, s: np.ndarray, x: float, y: float,
                  r_in: float, r_out: float, local_level: float) -> float:
    """RMS change introduced by compositing in an annulus clear of the object,
    relative to the local planet level (0 = compositing left it untouched)."""
    h, w = m.shape
    yy, xx = np.mgrid[0:h, 0:w]
    d2 = (xx - x) ** 2 + (yy - y) ** 2
    ann = (d2 >= r_in * r_in) & (d2 <= r_out * r_out)
    return float(np.sqrt(np.mean((m[ann] - s[ann]) ** 2)) / max(local_level, 1e-6))


# ── Pipeline plumbing ─────────────────────────────────────────────────────────

def _lum(path) -> np.ndarray:
    img = image_io.read_tif(path)
    return img if img.ndim == 2 else img.mean(axis=2).astype(np.float32)


def _run_step03(tif_dir: Path, out_dir: Path, filters: List[str]) -> Tuple[dict, PipelineConfig]:
    cache = out_dir / f"step03_cache_{'-'.join(filters)}.pkl"
    cfg = PipelineConfig(
        input_dir=tif_dir, output_base_dir=out_dir / "pipeline",
        filters=filters, save_step03=False,
        quality=QualityConfig(window_frames=3, cycle_minutes=225 / 60.0),
    )
    cfg.derotation.use_true_reprojection = True
    cfg.derotation.true_polar_equatorial_ratio = 0.9351  # Jupiter
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f), cfg
    res = quality_assess.run(cfg)
    with open(cache, "wb") as f:
        pickle.dump(res, f)
    return res, cfg


def _pick_windows(windows: List[dict], n: int) -> List[dict]:
    """Greedy non-overlapping windows, highest quality first, then time-ordered."""
    chosen: List[dict] = []
    for w in sorted(windows, key=lambda w: -w["window_quality"]):
        if all(w["window_end"] < c["window_start"] or w["window_start"] > c["window_end"] for c in chosen):
            chosen.append(w)
        if len(chosen) >= n:
            break
    return sorted(chosen, key=lambda w: w["center_time"])


def _derotate(rows: List[dict], t_ref: datetime, sess: dict, cfg: PipelineConfig):
    return derotation.derotate_filter(
        rows, t_ref, cfg.derotation.rotation_period_hours,
        warp_scale=1.0, align=True,
        min_quality_threshold=cfg.derotation.min_quality_threshold,
        pole_pa_deg=sess["pole_pa"], flip_direction=sess["derot_flip"],
        weight_power=1.0, use_true_reprojection=True,
        sub_observer_lat_deg=sess["B"], true_polar_equatorial_ratio=0.9351,
        flip_pole_axis=sess["flip_pole_axis"],
    )


def _surface_velocity(x: float, y: float, cx: float, cy: float, r: float,
                      sess: dict, cfg: PipelineConfig, sign: float) -> Tuple[float, float]:
    """Image-plane velocity (px/s) of the planetary surface point at (x, y)."""
    dx, dy = derotation._reprojection_point_shift(
        x, y, 60.0, cx, cy, r, cfg.derotation.rotation_period_hours,
        sess["B"], sess["pole_pa"], 0.9351, scale=1.0,
        flip_direction=sess["derot_flip"], flip_pole_axis=sess["flip_pole_axis"],
    )
    return sign * dx / 60.0, sign * dy / 60.0


def _sat_pos(x: float, y: float, cx: float, cy: float, r: float, name: str) -> SatellitePos:
    d = math.hypot(x - cx, y - cy)
    return SatellitePos(name=name, x_px=x, y_px=y, on_disk=d < r, dist_px=d)


# ── One injection experiment ──────────────────────────────────────────────────

def run_case(rows: List[dict], t_ref: datetime, P: np.ndarray, frame_centers: Dict[str, Tuple[float, float]],
             disk: Tuple[float, float, float], o_ref: Tuple[float, float], dv_r_per_h: float,
             obj_type: str, r_app: float, amplitude: float, sess: dict, cfg: PipelineConfig,
             tmp_dir: Path, sign: float, deltas_deg: List[float],
             noise_sigma: Optional[Dict[str, float]] = None, noise_mult: float = 1.0) -> List[dict]:
    cx, cy, R = disk
    x_ref, y_ref = cx + o_ref[0], cy + o_ref[1]
    vsx, vsy = _surface_velocity(x_ref, y_ref, cx, cy, R, sess, cfg, sign)
    vs = math.hypot(vsx, vsy)
    ex, ey = (vsx / vs, vsy / vs) if vs > 1e-9 else (1.0, 0.0)
    dv = dv_r_per_h * R / 3600.0  # px/s along the local drift direction
    vx, vy = vsx + dv * ex, vsy + dv * ey

    # Offsets from each frame's own disk centre at each frame's time.
    offsets: Dict[str, Tuple[float, float]] = {}
    inj_rows: List[dict] = []
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    for row in rows:
        dt = (row["timestamp"] - t_ref).total_seconds()
        ox, oy = o_ref[0] + vx * dt, o_ref[1] + vy * dt
        offsets[row["stem"]] = (ox, oy)
        fcx, fcy = frame_centers[row["stem"]]
        img = _inject(_lum(row["path"]), fcx + ox, fcy + oy, r_app, obj_type, amplitude)
        if noise_mult > 1.0 and noise_sigma:
            # Raise the total per-frame noise to noise_mult x the measured level
            # (independent per frame, fixed seed for reproducibility).
            seed = zlib.crc32(f"{row['stem']}|{obj_type}|{amplitude}|{dv_r_per_h}|{o_ref}|{noise_mult}".encode())
            sig = math.sqrt(noise_mult ** 2 - 1.0) * noise_sigma[row["stem"]]
            img = np.clip(img + np.random.default_rng(seed).normal(0.0, sig, img.shape), 0.0, 1.0).astype(np.float32)
        p = tmp_dir / Path(row["path"]).name
        image_io.write_tif_16bit(img, p)
        inj_rows.append({**row, "path": p})

    S, slog = _derotate(inj_rows, t_ref, sess, cfg)
    T = _inject(P, x_ref, y_ref, r_app, obj_type, amplitude)
    R_T = T - P
    r_ap = max(3.0 * r_app, r_app + 3 * _PSF_SIGMA_PX)
    local = float(np.median(P[int(y_ref) - 6:int(y_ref) + 7, int(x_ref) - 6:int(x_ref) + 7]))

    align = {f["stem"]: f.get("align_shift_px", [0.0, 0.0]) for f in slog.get("frames", [])}
    kept = {f["stem"] for f in slog.get("frames", []) if not f.get("outlier_excluded")}
    rows_sorted = sorted(
        [{**r, "align_shift_px": align.get(r["stem"], [0.0, 0.0])} for r in inj_rows if r["stem"] in kept],
        key=lambda r: r["timestamp"],
    )
    smear_px = math.hypot(vx - vsx, vy - vsy) * (
        (rows_sorted[-1]["timestamp"] - rows_sorted[0]["timestamp"]).total_seconds() if rows_sorted else 0.0)

    base = {"obj_type": obj_type, "dv_R_per_h": dv_r_per_h, "o_ref_x": o_ref[0], "o_ref_y": o_ref[1],
            "r_frac": math.hypot(*o_ref) / R, "r_app_px": r_app, "amplitude": amplitude,
            "disk_r_px": R, "n_frames": len(rows_sorted), "baseline_smear_px": smear_px,
            "noise_mult": noise_mult}
    out: List[dict] = []
    out.append({**base, "method": "single_rate", "delta_deg": 0.0,
                **_metrics(S - P, R_T, x_ref, y_ref, r_ap), "artifact_rms": 0.0})

    # Single-frame paste baseline (the current manual practice, in its best
    # form): the object is taken from the ONE frame closest to the reference
    # time and composited with the same capsule mask and Poisson blend, so the
    # only difference from multi-rate is the number of frames stacked.
    if rows_sorted:
        k = min(range(len(rows_sorted)),
                key=lambda i: abs((rows_sorted[i]["timestamp"] - t_ref).total_seconds()))
        rk = rows_sorted[k]
        pos_k = _sat_pos(cx + offsets[rk["stem"]][0], cy + offsets[rk["stem"]][1], cx, cy, R, "synthetic")
        ref0 = _sat_pos(x_ref, y_ref, cx, cy, R, "synthetic")
        single = _satellite_translate_stack([rk], [pos_k], ref0)
        traj0 = [(cx + offsets[r["stem"]][0], cy + offsets[r["stem"]][1]) for r in rows_sorted]
        M1 = _blend_poisson(S.copy(), single, ref0, r_app * _COVERAGE_SCALE, traj_xy=traj0, mask_shape="capsule")
        out.append({**base, "method": "single_frame_paste", "delta_deg": 0.0,
                    **_metrics(M1 - P, R_T, x_ref, y_ref, r_ap),
                    "artifact_rms": _artifact_rms(M1, S, x_ref, y_ref, r_ap + 2, r_ap + 12, local)})

    for delta in deltas_deg:
        c, s_ = math.cos(math.radians(delta)), math.sin(math.radians(delta))

        def rot(ox: float, oy: float) -> Tuple[float, float]:
            return cx + c * ox - s_ * oy, cy + s_ * ox + c * oy

        positions = [_sat_pos(*rot(*offsets[r["stem"]]), cx, cy, R, "synthetic") for r in rows_sorted]
        ref = _sat_pos(*rot(*o_ref), cx, cy, R, "synthetic")
        stack = _satellite_translate_stack(rows_sorted, positions, ref)
        traj = [(p.x_px, p.y_px) for p in positions if p.on_disk]
        M = _blend_poisson(S.copy(), stack, ref, r_app * _COVERAGE_SCALE, traj_xy=traj, mask_shape="capsule")
        out.append({**base, "method": "multi_rate", "delta_deg": delta,
                    **_metrics(M - P, R_T, x_ref, y_ref, r_ap),
                    "artifact_rms": _artifact_rms(M, S, x_ref, y_ref, r_ap + 2, r_ap + 12, local)})
        if delta == 0.0:
            _save_panel(P, S, M, T, x_ref, y_ref, r_ap, tmp_dir.parent / "panels",
                        f"{obj_type}_dv{dv_r_per_h:.2f}_r{base['r_frac']:.2f}_{rows[0]['stem'][:15]}.png")
    return out


def _save_panel(P, S, M, T, x, y, r_ap, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    half = int(r_ap * 3)
    ix, iy = int(round(x)), int(round(y))

    def crop(a):
        c = a[max(0, iy - half): iy + half, max(0, ix - half): ix + half]
        return cv2.resize(c, (c.shape[1] * 4, c.shape[0] * 4), interpolation=cv2.INTER_NEAREST)

    tiles = [crop(a) for a in (P, S, M, T)]
    lo, hi = np.percentile(np.concatenate([t.ravel() for t in tiles]), [1, 99.5])
    tiles = [np.clip((t - lo) / max(hi - lo, 1e-6), 0, 1) for t in tiles]
    row = np.concatenate([np.pad(t, 2, constant_values=1.0) for t in tiles], axis=1)
    img = (row * 255).astype(np.uint8)
    for i, lab in enumerate(["planet-only", "single-rate", "multi-rate", "truth"]):
        cv2.putText(img, lab, (8 + i * (tiles[0].shape[1] + 4), 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
    cv2.imwrite(str(out_dir / name), img)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tif-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--filters", nargs="+", default=["IR"])
    ap.add_argument("--n-windows", type=int, default=3)
    ap.add_argument("--dv", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.45],
                    help="Object-minus-surface velocity along the drift direction, in disk radii per hour")
    ap.add_argument("--r-fracs", nargs="+", type=float, default=[0.0, 0.5],
                    help="Distance of the reference position from disk centre (fraction of radius), "
                         "placed along +/- the drift axis")
    ap.add_argument("--deltas", nargs="+", type=float, default=[0.0, 2.0, 4.0, 8.0, 12.0])
    ap.add_argument("--r-app", type=float, default=3.0, help="Apparent object radius (px)")
    ap.add_argument("--sat-amp", nargs="+", type=float, default=[0.15])
    ap.add_argument("--shadow-amp", nargs="+", type=float, default=[0.85])
    ap.add_argument("--noise-mult", nargs="+", type=float, default=[1.0],
                    help="Total per-frame noise as a multiple of the measured level (1 = data as is)")
    ap.add_argument("--results-name", default="results.csv")
    ap.add_argument("--quick", action="store_true", help="Sign check only")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    res03, cfg = _run_step03(args.tif_dir, args.out_dir, args.filters)
    windows = res03["windows"]
    sess_cache = args.out_dir / "session_params.json"
    if sess_cache.exists():
        sess = json.loads(sess_cache.read_text())
    else:
        pole_pa = derotate_stack._scan_session_pole_pa(res03.get("scores", {}), cfg) or 0.0
        derot_flip, _, _ = derotate_stack._detect_session_flip_ns(windows, cfg, pole_pa)
        fpa = derotate_stack._detect_session_pole_axis_flip(windows, cfg, pole_pa, derot_flip)
        sess = {"pole_pa": float(pole_pa), "derot_flip": bool(derot_flip), "flip_pole_axis": bool(fpa)}
        sess_cache.write_text(json.dumps(sess, indent=2))
    print(f"[session] {sess}")

    chosen = _pick_windows(windows, args.n_windows)
    all_rows: List[dict] = []
    sign: Optional[float] = None
    for wi, win in enumerate(chosen):
        t_ref = win["center_time"]
        B = derotation.query_horizons_sub_observer_lat(
            horizons_id="599", t_utc=t_ref, observer_code=cfg.derotation.observer_code) or 0.0
        sess_w = {**sess, "B": float(B)}
        for filt in args.filters:
            rows = win["per_filter"].get(filt, {}).get("included", [])
            if len(rows) < 2:
                continue
            print(f"\n=== window {wi + 1} {t_ref:%H:%M} {filt}  ({len(rows)} frames, B={B:.2f}) ===")
            P, plog = _derotate(rows, t_ref, sess_w, cfg)
            cx, cy, R, _, _ = derotation.find_disk_center(P)
            centers, noise_sigma = {}, {}
            for r in rows:
                im = _lum(r["path"])
                fcx, fcy, fr, *_ = derotation.find_disk_center(im)
                centers[r["stem"]] = (fcx, fcy)
                # Pixel-scale residual (robust sigma) inside 0.6 R: an upper bound on the frame noise.
                hp = im - cv2.GaussianBlur(im, (0, 0), 1.5)
                yy, xx = np.mgrid[0:im.shape[0], 0:im.shape[1]]
                v = hp[(xx - fcx) ** 2 + (yy - fcy) ** 2 < (0.6 * fr) ** 2]
                noise_sigma[r["stem"]] = float(1.4826 * np.median(np.abs(v - np.median(v))))
            tmp = args.out_dir / "tmp_inject"

            # Drift direction at the centre, used to lay out reference positions.
            vx, vy = _surface_velocity(cx, cy, cx, cy, R, sess_w, cfg, 1.0)
            n = math.hypot(vx, vy)
            ux, uy = (vx / n, vy / n) if n > 0 else (1.0, 0.0)

            if sign is None:
                # A surface-locked object (dv=0) must be recovered by the planet
                # de-rotation alone; pick the velocity sign convention that does so.
                best = None
                for sg in (1.0, -1.0):
                    rr = run_case(rows, t_ref, P, centers, (cx, cy, R), (0.0, 0.0), 0.0, "shadow",
                                  args.r_app, args.shadow_amp[0], sess_w, cfg, tmp, sg, [])
                    fr = rr[0]["flux_recovery"]
                    print(f"[sign check] sign={sg:+.0f}: single-rate flux_recovery={fr:.3f} "
                          f"elongation={rr[0]['elongation']:.2f}")
                    if best is None or abs(fr - 1.0) < abs(best[1] - 1.0):
                        best = (sg, fr)
                sign = best[0]
                print(f"[sign check] using sign={sign:+.0f}")
                if args.quick:
                    return

            for frac in args.r_fracs:
                for side in ((1.0,) if frac == 0 else (1.0, -1.0)):
                    o_ref = (side * frac * R * ux, side * frac * R * uy)
                    objs = [("satellite", a) for a in args.sat_amp] + [("shadow", a) for a in args.shadow_amp]
                    for (obj_type, amp), nm, dv in [(o, nm, dv) for o in objs for nm in args.noise_mult for dv in args.dv]:
                        rr = run_case(rows, t_ref, P, centers, (cx, cy, R), o_ref, dv, obj_type,
                                      args.r_app, amp, sess_w, cfg, tmp, sign, args.deltas,
                                      noise_sigma=noise_sigma, noise_mult=nm)
                        for r in rr:
                            r.update(window=wi + 1, filter=filt, t_ref=f"{t_ref:%Y-%m-%dT%H:%M}")
                        all_rows.extend(rr)
                        s0 = rr[0]
                        m0 = next(r for r in rr if r["method"] == "multi_rate" and r["delta_deg"] == 0.0)
                        print(f"  {obj_type:9s} r={frac:.1f}{'+' if side > 0 else '-'} dv={dv:.2f}R/h "
                              f"smear={s0['baseline_smear_px']:.1f}px | flux S={s0['flux_recovery']:.3f} "
                              f"M={m0['flux_recovery']:.3f} | nrmse S={s0['nrmse']:.3f} M={m0['nrmse']:.3f}")
            # Write incrementally so a long run is never lost.
            _write_csv(all_rows, args.out_dir / args.results_name)
    _write_csv(all_rows, args.out_dir / args.results_name)
    print(f"\nWrote {len(all_rows)} rows → {args.out_dir / args.results_name}")


def _write_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
