"""
Re-measure the real transit events with a matched-filter contrast on the RAW
(unsharpened) Step 4 stacks, alongside the paper's original extremum-based
aperture contrast (raw and wavelet-sharpened) for continuity.

Expects the output directories written by
    python aperture_contrast.py --tif-dir <session>/TIFs --output-dir <OUT>/<session>
i.e. <OUT>/<session>/{baseline,composite}/step04_derotated/window_XX/.

Object positions, non-overlapping window selection, apparent radii and the
"compositing was applied here" test are taken from aperture_contrast.py so the
measurement locations match the original analysis exactly.

Matched-filter contrast (per object, per filter, per window):
    D   = I - GaussianBlur(I, 4 r_app)          (removes the planetary gradient)
    Rmf = GaussianBlur(D, r_app)                (matched to the object scale)
    p*  = location of the object-signed extremum of Rmf, searched within one
          aperture radius of the predicted position on the MEAN of the baseline
          and composite images, then used for BOTH (so neither image is favoured
          by a per-image peak search)
    C   = sign * Rmf(p*) / B,  B = median of I in the 1.5-2.5 r_ref annulus
This is a Gaussian-weighted local mean rather than a single-pixel extremum, and
needs no wavelet sharpening.

Usage:
    python validation/real_events_metrics.py D:/Astro/real_events_v2 --out D:/Astro/real_events_v2/summary
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aperture_contrast as ac  # noqa: E402
from pipeline.modules.derotation import find_disk_center, refine_disk_radius_for_plate_scale  # noqa: E402
from pipeline.modules.image_io import read_tif  # noqa: E402
from pipeline.modules.satellite_tracker import SatelliteTracker  # noqa: E402

_R_SCALE = 2.0  # aperture reference radius = apparent diameter (aperture_contrast.py convention)


def _mf_map(img: np.ndarray, r_app: float) -> np.ndarray:
    s_t = max(r_app, 1.0)
    hp = img - cv2.GaussianBlur(img, (0, 0), 4.0 * s_t)
    return cv2.GaussianBlur(hp, (0, 0), s_t)


def _mf_contrast(img_b: np.ndarray, img_c: np.ndarray, x: float, y: float, r_app: float,
                 r_ref: float, sign: float):
    mb, mc = _mf_map(img_b, r_app), _mf_map(img_c, r_app)
    h, w = img_b.shape
    yy, xx = np.mgrid[0:h, 0:w]
    search = (xx - x) ** 2 + (yy - y) ** 2 <= r_ref * r_ref
    avg = sign * 0.5 * (mb + mc)
    avg_s = np.where(search, avg, -np.inf)
    iy, ix = np.unravel_index(int(np.argmax(avg_s)), avg_s.shape)
    ann = ac._annular_mask(img_b.shape, ix, iy, 1.5 * r_ref, 2.5 * r_ref)
    bb, bc = float(np.median(img_b[ann])), float(np.median(img_c[ann]))
    noise = float(np.std(mb[ann]))
    cb = sign * float(mb[iy, ix]) / max(bb, 1e-6)
    cc = sign * float(mc[iy, ix]) / max(bc, 1e-6)
    snr_b = sign * float(mb[iy, ix]) / max(noise, 1e-9)
    return cb, cc, snr_b, float(np.hypot(ix - x, iy - y))


def measure_session(sess_dir: Path) -> List[dict]:
    base_root = sess_dir / "baseline" / "step04_derotated"
    comp_root = sess_dir / "composite" / "step04_derotated"
    if not comp_root.is_dir():
        return []
    rows: List[dict] = []
    for comp_win in ac._deoverlap_win_dirs(sorted(comp_root.glob("window_*"))):
        base_win = base_root / comp_win.name
        log_path = comp_win / "derotation_log.json"
        if not base_win.is_dir() or not log_path.exists():
            continue
        log = json.loads(log_path.read_text())
        t_c = datetime.strptime(log["center_time"], "%Y-%m-%dT%H:%M:%SZ")
        sat_log, ses_log = log.get("satellite", {}), log.get("session", {})
        if not sat_log:
            continue
        pole_pa = float(sat_log.get("pole_pa_deg", ses_log.get("pole_pa_deg", 0.0)))
        np_ang = float(sat_log.get("np_ang_deg", 0.0))
        flip_ns = bool(sat_log.get("tracker_flip_ns", ses_log.get("tracker_flip_ns", False)))
        tracker = SatelliteTracker(flip_ew=bool(ses_log.get("flip_ew", False)), flip_ns=flip_ns)

        for comp_tif in sorted(comp_win.glob("*_derotated.tif")):
            filt = comp_tif.stem.replace("_derotated", "")
            base_tif = base_win / comp_tif.name
            if not base_tif.exists():
                continue
            raw_c, raw_b = read_tif(comp_tif), read_tif(base_tif)
            if raw_b.ndim == 3:
                raw_b, raw_c = raw_b.mean(axis=2), raw_c.mean(axis=2)
            stored = sat_log.get("disk_centers", {}).get(filt)
            if stored:
                cx, cy, r = float(stored["cx"]), float(stored["cy"]), float(stored["r"])
                r_b, ang = r, 0.0
            else:
                cx, cy, r, r_b, ang = find_disk_center(raw_b)
            sh_b = ac._apply_wavelet_display(raw_b, cx, cy, r)
            sh_c = ac._apply_wavelet_display(raw_c, cx, cy, r)
            r_ps, _, _ = refine_disk_radius_for_plate_scale(sh_b, cx, cy, r, r_b, ang)
            ps = tracker.get_plate_scale(r_ps, t_c)
            body = tracker.get_positions([t_c], cx, cy, r_ps, pole_pa_deg=pole_pa, np_ang_deg=np_ang)
            try:
                shad = tracker.get_shadow_positions([t_c], cx, cy, r_ps, pole_pa_deg=pole_pa, np_ang_deg=np_ang) or {}
            except Exception:
                shad = {}
            objs = [(n, pl, "satellite", n) for n, pl in body.items()] + \
                   [(n, pl, "shadow", n.replace("_shadow", "")) for n, pl in shad.items()]
            for name, plist, otype, moon in objs:
                p = plist[0] if plist else None
                if p is None or not p.on_disk:
                    continue
                app_r = ac._apparent_radius_px(moon, t_c, ps)
                r_ref = max(app_r * _R_SCALE, 3.0)
                mask = ac._annular_mask(raw_b.shape, p.x_px, p.y_px, 0.0, 2.5 * r_ref)
                if np.allclose(raw_b[mask], raw_c[mask], atol=1e-6):
                    continue  # compositing not applied at this object in this filter
                sign = 1.0 if otype == "satellite" else -1.0
                cb, cc, snr_b, loc_off = _mf_contrast(raw_b, raw_c, p.x_px, p.y_px, app_r, r_ref, sign)
                rows.append({
                    "session": sess_dir.name, "window": comp_win.name, "filter": filt,
                    "obj_type": otype, "object": name, "app_r_px": app_r,
                    "mf_base": cb, "mf_comp": cc, "mf_snr_base": snr_b, "mf_loc_offset_px": loc_off,
                    "ext_raw_base": ac.measure_aperture_contrast(raw_b, p.x_px, p.y_px, r_ref, otype),
                    "ext_raw_comp": ac.measure_aperture_contrast(raw_c, p.x_px, p.y_px, r_ref, otype),
                    "ext_sharp_base": ac.measure_aperture_contrast(sh_b, p.x_px, p.y_px, r_ref, otype),
                    "ext_sharp_comp": ac.measure_aperture_contrast(sh_c, p.x_px, p.y_px, r_ref, otype),
                })
    return rows


def _pct(c: pd.Series, b: pd.Series) -> pd.Series:
    return 100.0 * (c - b) / b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    for sess in sorted(p for p in args.root.iterdir() if p.is_dir() and p.name.isdigit()):
        print(f"[{sess.name}]")
        rows.extend(measure_session(sess))
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "real_events_metrics.csv", index=False)
    if df.empty:
        print("no measurements")
        return

    df["d_mf"] = _pct(df["mf_comp"], df["mf_base"])
    df["d_ext_raw"] = _pct(df["ext_raw_comp"], df["ext_raw_base"])
    df["d_ext_sharp"] = _pct(df["ext_sharp_comp"], df["ext_sharp_base"])
    # The paper's 0.05 minimum-baseline gate, applied to each metric's own baseline.
    gate_sharp = df["ext_sharp_base"] >= 0.05
    gate_raw = df["ext_raw_base"] >= 0.05

    out = []
    for (otype, filt), g in df.groupby(["obj_type", "filter"]):
        gs = g[gate_sharp.loc[g.index]]
        gr = g[gate_raw.loc[g.index]]
        det = g[g["mf_snr_base"] >= 3.0]
        out.append({
            "obj_type": otype, "filter": filt, "N": len(g),
            "N_sharp": len(gs), "sharp_ext_mean%": round(gs["d_ext_sharp"].mean(), 1),
            "N_raw": len(gr), "raw_ext_mean%": round(gr["d_ext_raw"].mean(), 1),
            "mf_ratio_of_means%": round(_pct(g["mf_comp"].mean(), g["mf_base"].mean()), 1),
            "mf_median%": round(g["d_mf"].median(), 1),
            "mf_frac_positive": round((g["mf_comp"] > g["mf_base"]).mean(), 2),
            "N_det3": len(det),
            "mf_det3_median%": round(det["d_mf"].median(), 1) if len(det) else np.nan,
        })
    t = pd.DataFrame(out)
    t.to_csv(args.out / "real_events_summary.csv", index=False)
    per_sess = df.groupby(["obj_type", "session", "object"]).apply(
        lambda g: pd.Series({"N": len(g), "mf_median%": g["d_mf"].median(),
                             "mf_frac_positive": (g["mf_comp"] > g["mf_base"]).mean()})).round(2)
    per_sess.to_csv(args.out / "real_events_per_session.csv")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(t.to_string(index=False))
        print()
        print(per_sess.to_string())


if __name__ == "__main__":
    main()
