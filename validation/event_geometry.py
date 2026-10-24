"""
Per-event geometry of the real transits: relative speed, predicted smear and
predicted contrast gain, compared with the measured gain; and the along-path /
cross-path decomposition of the position offsets found by the opt-in CV
position refinement.

For every non-overlapping window, filter and on-disk object (same selection as
aperture_contrast.py / real_events_metrics.py):
    v_obj   predicted image velocity of the object (tracker positions at t_c and
            t_c + 60 s), giving the path direction e and its normal n
    z       line-of-sight coordinate of the surface point under the object on a
            flattened globe (axis ratio 0.9351), with the equator taken along e
    dv      v_obj . e - omega * z, in Jupiter radii per hour (omega = 2 pi / P)
    smear   |dv| * R * (last - first image time of that filter in the window)
    gain    predicted multi-rate / single-rate peak ratio - 1 for an object
            modelled as a uniform disk of the object's apparent radius, blurred
            by a Gaussian, stacked at the window's image times and weights.
            Two blurs: sigma = 1.2 px (the artificial-object test) and the
            diffraction limit of a 235-mm aperture at the filter wavelength.
Measured gains are the unsharpened extremum-aperture values of
real_events_metrics.csv (paper configuration).

Position offsets: for every object/filter/window where the CV refinement run
applied an offset (detected minus predicted), the offset is projected on e
(along-path, positive = ahead of the prediction) and n (cross-path), given in px,
in units of the band width sigma_perp = 1.5 r_app, and, for the along-path
part, as the equivalent time shift (minutes).  Diagnostic only: nothing is fed
back into the pipeline.

Usage:
    python validation/event_geometry.py D:/Astro/real_events_v2 D:/Astro/real_events_v2_cvref \
        --metrics D:/Astro/real_events_v2/summary/real_events_metrics.csv \
        --out D:/Astro/real_events_v2/summary
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aperture_contrast as ac  # noqa: E402
from pipeline.modules.derotation import find_disk_center, refine_disk_radius_for_plate_scale  # noqa: E402
from pipeline.modules.image_io import read_tif  # noqa: E402
from pipeline.modules.satellite_tracker import SatelliteTracker  # noqa: E402

_R_SCALE = 2.0
_COVERAGE_SCALE = 1.5
_AXIS_RATIO = 0.9351
_APERTURE_M = 0.235
_WAVELENGTH_UM = {"IR": 0.75, "R": 0.63, "G": 0.53, "B": 0.45, "CH4": 0.889}
_TEST_SIGMA_PX = 1.2


def _lum(path: Path) -> np.ndarray:
    img = read_tif(path)
    return img if img.ndim == 2 else img.mean(axis=2)


def _peak_ratio(a: float, sigma: float, shifts: List[float], weights: List[float]) -> float:
    """Peak of the weighted stack of shifted copies over the peak of one copy."""
    pad = 4.0 * sigma + a + max(abs(s) for s in shifts) + 2.0
    step = 0.1
    xs = np.arange(-pad, pad + step, step)
    xx, yy = np.meshgrid(xs, xs)
    disk = ((xx ** 2 + yy ** 2) <= a * a).astype(float)
    ksz = int(2 * math.ceil(4 * sigma / step) + 1)
    one = cv2.GaussianBlur(disk, (ksz, ksz), sigma / step)
    w = np.asarray(weights, float)
    w = w / w.sum()
    stack = np.zeros_like(one)
    for s, wi in zip(shifts, w):
        m = np.float32([[1, 0, s / step], [0, 1, 0]])
        stack += wi * cv2.warpAffine(one, m, (one.shape[1], one.shape[0]), flags=cv2.INTER_LINEAR)
    return float(stack.max() / one.max())


def _objects(tracker, t, cx, cy, r_ps, pole_pa, np_ang):
    body = tracker.get_positions([t], cx, cy, r_ps, pole_pa_deg=pole_pa, np_ang_deg=np_ang)
    try:
        shad = tracker.get_shadow_positions([t], cx, cy, r_ps, pole_pa_deg=pole_pa, np_ang_deg=np_ang) or {}
    except Exception:
        shad = {}
    out = {n: ("satellite", n, pl[0] if pl else None) for n, pl in body.items()}
    out.update({n: ("shadow", n.replace("_shadow", ""), pl[0] if pl else None) for n, pl in shad.items()})
    return out


def measure(ephem_sess: Path, ref_sess: Optional[Path]) -> List[dict]:
    base_root = ephem_sess / "baseline" / "step04_derotated"
    comp_root = ephem_sess / "composite" / "step04_derotated"
    rows: List[dict] = []
    if not comp_root.is_dir():
        return rows
    for comp_win in ac._deoverlap_win_dirs(sorted(comp_root.glob("window_*"))):
        base_win = base_root / comp_win.name
        if not base_win.is_dir():
            continue
        log = json.loads((comp_win / "derotation_log.json").read_text())
        sat_log, ses_log = log.get("satellite", {}), log.get("session", {})
        if not sat_log:
            continue
        rcent: Dict = {}
        rlog_path = ref_sess / "composite" / "step04_derotated" / comp_win.name / "derotation_log.json" if ref_sess else None
        if rlog_path is not None and rlog_path.exists():
            rcent = json.loads(rlog_path.read_text()).get("satellite", {}).get("disk_centers", {})
        sat_ref = rcent.get("_satellite_refinement", {}) or {}
        t_c = datetime.strptime(log["center_time"], "%Y-%m-%dT%H:%M:%SZ")
        pole_pa = float(sat_log.get("pole_pa_deg", ses_log.get("pole_pa_deg", 0.0)))
        np_ang = float(sat_log.get("np_ang_deg", 0.0))
        tracker = SatelliteTracker(flip_ew=bool(ses_log.get("flip_ew", False)),
                                   flip_ns=bool(sat_log.get("tracker_flip_ns", ses_log.get("tracker_flip_ns", False))))
        for comp_tif in sorted(comp_win.glob("*_derotated.tif")):
            filt = comp_tif.stem.replace("_derotated", "")
            b_path = base_win / comp_tif.name
            finfo = log.get("filters", {}).get(filt, {})
            if not b_path.exists() or not finfo.get("frames"):
                continue
            img_b, img_e = _lum(b_path), _lum(comp_tif)
            stored = sat_log.get("disk_centers", {}).get(filt)
            if stored:
                cx, cy, r = float(stored["cx"]), float(stored["cy"]), float(stored["r"])
                rb, ang = r, 0.0
            else:
                cx, cy, r, rb, ang = find_disk_center(img_b)
            r_ps, _, _ = refine_disk_radius_for_plate_scale(img_b, cx, cy, r, rb, ang)
            ps = tracker.get_plate_scale(r_ps, t_c)
            now = _objects(tracker, t_c, cx, cy, r_ps, pole_pa, np_ang)
            later = _objects(tracker, t_c + timedelta(seconds=60), cx, cy, r_ps, pole_pa, np_ang)
            frames = finfo["frames"]
            dts = [float(f["dt_sec"]) for f in frames]
            wts = [max(float(f.get("norm_score", 1.0)), 1e-6) for f in frames]
            omega = 2.0 * math.pi / float(finfo.get("period_hours", 9.9281))  # rad/h
            shad_ref = (rcent.get(filt, {}) or {}).get("shadow_refinement", {}) or {}
            for name, (otype, moon, p) in now.items():
                q = later.get(name, (None, None, None))[2]
                if p is None or q is None or not p.on_disk:
                    continue
                app_r = ac._apparent_radius_px(moon, t_c, ps)
                r_ref = max(app_r * _R_SCALE, 3.0)
                mask = ac._annular_mask(img_b.shape, p.x_px, p.y_px, 0.0, 2.5 * r_ref)
                if np.allclose(img_b[mask], img_e[mask], atol=1e-6):
                    continue  # no compositing at this object/filter
                vx, vy = q.x_px - p.x_px, q.y_px - p.y_px  # px/min
                speed = math.hypot(vx, vy)
                ex, ey = vx / speed, vy / speed
                nx, ny = -ey, ex
                X = ((p.x_px - cx) * ex + (p.y_px - cy) * ey) / r_ps
                Y = ((p.x_px - cx) * nx + (p.y_px - cy) * ny) / (r_ps * _AXIS_RATIO)
                z = math.sqrt(max(0.0, 1.0 - X * X - Y * Y))
                v_obj = speed * 60.0 / r_ps  # R_J/h
                dv = v_obj - omega * z
                span_h = (max(dts) - min(dts)) / 3600.0
                smear = abs(dv) * r_ps * span_h
                shifts = [dv * r_ps * dt / 3600.0 for dt in dts]
                fwhm_px = 1.03 * _WAVELENGTH_UM.get(filt, 0.7) * 1e-6 / _APERTURE_M * 206265.0 / ps
                sig_diff = fwhm_px / 2.3548
                g_test = 1.0 / _peak_ratio(3.0, _TEST_SIGMA_PX, shifts, wts) - 1.0
                g_obj = 1.0 / _peak_ratio(app_r, _TEST_SIGMA_PX, shifts, wts) - 1.0
                g_diff = 1.0 / _peak_ratio(app_r, sig_diff, shifts, wts) - 1.0
                refinfo = sat_ref if otype == "satellite" else shad_ref.get(name, {})
                applied = bool(refinfo.get("applied"))
                odx, ody = (refinfo.get("offset_px") or [math.nan, math.nan]) if applied else (math.nan, math.nan)
                along = odx * ex + ody * ey
                cross = odx * nx + ody * ny
                sig_perp = app_r * _COVERAGE_SCALE
                rows.append({
                    "session": ephem_sess.name, "window": comp_win.name, "filter": filt,
                    "obj_type": otype, "object": name, "app_r_px": app_r, "R_px": r_ps,
                    "x_frac": X, "y_frac": Y, "r_frac": math.hypot(p.x_px - cx, p.y_px - cy) / r_ps,
                    "v_obj": v_obj, "v_surf": omega * z, "dv": dv,
                    "span_min": span_h * 60.0, "smear_px": smear,
                    "fwhm_diff_px": fwhm_px,
                    "gain_test": g_test, "gain_obj_test_blur": g_obj, "gain_obj_diff_blur": g_diff,
                    "cv_applied": applied, "cv_confidence": refinfo.get("confidence"),
                    "off_dx": odx, "off_dy": ody, "off_px": math.hypot(odx, ody),
                    "off_along_px": along, "off_cross_px": cross,
                    "off_over_sigma": math.hypot(odx, ody) / sig_perp,
                    "along_over_sigma": along / sig_perp, "cross_over_sigma": cross / sig_perp,
                    "off_angle_to_motion_deg": math.degrees(math.atan2(cross, along)),
                    "along_time_min": along / speed,
                    "speed_px_per_min": speed,
                })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ephem_root", type=Path)
    ap.add_argument("ref_root", type=Path)
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-contrast", type=float, default=0.05)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    for sess in sorted(p for p in args.ephem_root.iterdir() if p.is_dir() and p.name.isdigit()):
        print(f"[{sess.name}]")
        rows.extend(measure(sess, args.ref_root / sess.name))
    geo = pd.DataFrame(rows)
    geo.to_csv(args.out / "event_geometry.csv", index=False)

    met = pd.read_csv(args.metrics, dtype={"session": str})
    geo["session"] = geo["session"].astype(str)
    df = geo.merge(met[["session", "window", "filter", "object", "ext_raw_base", "ext_raw_comp"]],
                   on=["session", "window", "filter", "object"], how="left")
    df = df[~((df["session"] == "260328") & (df["obj_type"] == "shadow"))]
    df = df[df["ext_raw_base"] >= args.min_contrast]
    df["gain_meas"] = df["ext_raw_comp"] / df["ext_raw_base"] - 1.0

    pd.set_option("display.width", 200)
    print("\n=== Per event (unsharpened, windows x filters pooled) ===")
    agg = df.groupby(["obj_type", "session", "object"]).agg(
        n=("gain_meas", "size"),
        x_min=("x_frac", lambda s: s.abs().min()), x_max=("x_frac", lambda s: s.abs().max()),
        r_min=("r_frac", "min"), r_max=("r_frac", "max"),
        dv=("dv", "mean"), dv_min=("dv", "min"), dv_max=("dv", "max"),
        smear=("smear_px", "mean"), smear_min=("smear_px", "min"), smear_max=("smear_px", "max"),
        g_test=("gain_test", "mean"), g_obj=("gain_obj_test_blur", "mean"),
        g_diff=("gain_obj_diff_blur", "mean"), g_meas=("gain_meas", "mean"),
    )
    print(agg.round(3).to_string())
    agg.to_csv(args.out / "event_predicted_vs_measured.csv")

    print("\n=== Per event, by filter: measured vs predicted (diffraction blur) ===")
    byf = df.groupby(["obj_type", "session", "object", "filter"]).agg(
        n=("gain_meas", "size"), smear=("smear_px", "mean"),
        g_obj=("gain_obj_test_blur", "mean"), g_diff=("gain_obj_diff_blur", "mean"),
        g_meas=("gain_meas", "mean"))
    print(byf.round(3).to_string())
    byf.to_csv(args.out / "event_predicted_vs_measured_by_filter.csv")

    off = geo[geo["cv_applied"]].copy()
    # The satellite refinement is one CH4-based offset per window, shared by all filters.
    sat = off[off["obj_type"] == "satellite"].drop_duplicates(["session", "window", "object"])
    shd = off[off["obj_type"] == "shadow"]
    off = pd.concat([sat, shd])
    off.to_csv(args.out / "position_offsets.csv", index=False)
    print("\n=== Position offsets (CV refinement, detected - predicted) ===")
    cols = ["session", "window", "filter", "obj_type", "object", "off_px", "off_along_px", "off_cross_px",
            "off_over_sigma", "along_over_sigma", "cross_over_sigma", "off_angle_to_motion_deg",
            "along_time_min", "r_frac"]
    print(off[cols].round(2).to_string(index=False))
    print("\n=== Offset summary by object ===")
    summ = off.groupby(["obj_type", "session", "object"]).agg(
        n=("off_px", "size"), n_windows=("window", "nunique"),
        off=("off_px", "median"), off_max=("off_px", "max"),
        along=("off_along_px", "median"), cross=("off_cross_px", "median"),
        cross_abs_max=("off_cross_px", lambda s: s.abs().max()),
        off_sig=("off_over_sigma", "median"), off_sig_max=("off_over_sigma", "max"),
        cross_sig=("cross_over_sigma", lambda s: s.abs().median()),
        ang=("off_angle_to_motion_deg", "median"),
        t_min=("along_time_min", "min"), t_med=("along_time_min", "median"), t_max=("along_time_min", "max"),
    )
    print(summ.round(2).to_string())
    summ.to_csv(args.out / "position_offsets_summary.csv")


if __name__ == "__main__":
    main()
