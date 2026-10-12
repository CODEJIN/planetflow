"""
Effect of the opt-in CV position refinement on the real transit events.

Compares, for every on-disk satellite/shadow, filter and non-overlapping window:
    baseline       planet-only Step 4 stack
    comp_ephem     multi-rate composite at ephemeris-predicted positions (default)
    comp_refined   multi-rate composite with cv_position_refinement_enabled=True
Both composites are measured with the raw (unsharpened) extremum aperture
contrast at the SAME location: the refined position where refinement was
applied (the best available estimate of the true position), otherwise the
predicted position.

Inputs are the output directories of aperture_contrast.run_from_tif_dir():
    <EPHEM_ROOT>/<session>/{baseline,composite}/step04_derotated/window_XX/
    <REF_ROOT>/<session>/composite/step04_derotated/window_XX/

Usage:
    python validation/cv_refinement_effect.py D:/Astro/real_events_v2 D:/Astro/real_events_v2_cvref \
        --out D:/Astro/real_events_v2_cvref/summary
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aperture_contrast as ac  # noqa: E402
from pipeline.modules.derotation import find_disk_center, refine_disk_radius_for_plate_scale  # noqa: E402
from pipeline.modules.image_io import read_tif  # noqa: E402
from pipeline.modules.satellite_tracker import SatelliteTracker  # noqa: E402

_R_SCALE = 2.0
_COVERAGE_SCALE = 1.5


def _lum(path: Path) -> np.ndarray:
    img = read_tif(path)
    return img if img.ndim == 2 else img.mean(axis=2)


def measure(ephem_sess: Path, ref_sess: Path) -> List[dict]:
    base_root = ephem_sess / "baseline" / "step04_derotated"
    comp_root = ephem_sess / "composite" / "step04_derotated"
    ref_root = ref_sess / "composite" / "step04_derotated"
    rows: List[dict] = []
    if not comp_root.is_dir() or not ref_root.is_dir():
        return rows
    for comp_win in ac._deoverlap_win_dirs(sorted(comp_root.glob("window_*"))):
        base_win, ref_win = base_root / comp_win.name, ref_root / comp_win.name
        if not (base_win.is_dir() and ref_win.is_dir() and (ref_win / "derotation_log.json").exists()):
            continue
        log = json.loads((comp_win / "derotation_log.json").read_text())
        rlog = json.loads((ref_win / "derotation_log.json").read_text())
        sat_log, ses_log = log.get("satellite", {}), log.get("session", {})
        if not sat_log:
            continue
        rcent = rlog.get("satellite", {}).get("disk_centers", {})
        sat_ref = rcent.get("_satellite_refinement", {}) or {}
        t_c = datetime.strptime(log["center_time"], "%Y-%m-%dT%H:%M:%SZ")
        pole_pa = float(sat_log.get("pole_pa_deg", ses_log.get("pole_pa_deg", 0.0)))
        np_ang = float(sat_log.get("np_ang_deg", 0.0))
        tracker = SatelliteTracker(flip_ew=bool(ses_log.get("flip_ew", False)),
                                   flip_ns=bool(sat_log.get("tracker_flip_ns", ses_log.get("tracker_flip_ns", False))))
        for comp_tif in sorted(comp_win.glob("*_derotated.tif")):
            filt = comp_tif.stem.replace("_derotated", "")
            b_path, r_path = base_win / comp_tif.name, ref_win / comp_tif.name
            if not (b_path.exists() and r_path.exists()):
                continue
            img_b, img_e, img_r = _lum(b_path), _lum(comp_tif), _lum(r_path)
            stored = sat_log.get("disk_centers", {}).get(filt)
            if stored:
                cx, cy, r = float(stored["cx"]), float(stored["cy"]), float(stored["r"])
                rb, ang = r, 0.0
            else:
                cx, cy, r, rb, ang = find_disk_center(img_b)
            r_ps, _, _ = refine_disk_radius_for_plate_scale(img_b, cx, cy, r, rb, ang)
            ps = tracker.get_plate_scale(r_ps, t_c)
            body = tracker.get_positions([t_c], cx, cy, r_ps, pole_pa_deg=pole_pa, np_ang_deg=np_ang)
            try:
                shad = tracker.get_shadow_positions([t_c], cx, cy, r_ps, pole_pa_deg=pole_pa, np_ang_deg=np_ang) or {}
            except Exception:
                shad = {}
            shad_ref = (rcent.get(filt, {}) or {}).get("shadow_refinement", {}) or {}
            objs = [(n, pl, "satellite", n, sat_ref) for n, pl in body.items()] + \
                   [(n, pl, "shadow", n.replace("_shadow", ""), shad_ref.get(n, {})) for n, pl in shad.items()]
            for name, plist, otype, moon, refinfo in objs:
                p = plist[0] if plist else None
                if p is None or not p.on_disk:
                    continue
                app_r = ac._apparent_radius_px(moon, t_c, ps)
                r_ref = max(app_r * _R_SCALE, 3.0)
                mask = ac._annular_mask(img_b.shape, p.x_px, p.y_px, 0.0, 2.5 * r_ref)
                if np.allclose(img_b[mask], img_e[mask], atol=1e-6):
                    continue  # no compositing at this object/filter
                applied = bool(refinfo.get("applied"))
                dx, dy = (refinfo.get("offset_px") or [0.0, 0.0]) if applied else (0.0, 0.0)
                x, y = p.x_px + dx, p.y_px + dy
                cb = ac.measure_aperture_contrast(img_b, x, y, r_ref, otype)
                ce = ac.measure_aperture_contrast(img_e, x, y, r_ref, otype)
                cr = ac.measure_aperture_contrast(img_r, x, y, r_ref, otype)
                rows.append({
                    "session": ephem_sess.name, "window": comp_win.name, "filter": filt,
                    "obj_type": otype, "object": name, "attempted": bool(refinfo.get("attempted")),
                    "applied": applied, "confidence": refinfo.get("confidence"),
                    "offset_px": math.hypot(dx, dy),
                    "offset_over_sigma": math.hypot(dx, dy) / (app_r * _COVERAGE_SCALE),
                    "c_base": cb, "c_ephem": ce, "c_refined": cr,
                    "identical_composites": bool(np.allclose(img_e, img_r, atol=1e-6)),
                })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ephem_root", type=Path)
    ap.add_argument("ref_root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    for sess in sorted(p for p in args.ephem_root.iterdir() if p.is_dir() and p.name.isdigit()):
        print(f"[{sess.name}]")
        rows.extend(measure(sess, args.ref_root / sess.name))
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "cv_refinement_effect.csv", index=False)
    if df.empty:
        print("no measurements")
        return
    df["d_ephem"] = 100 * (df["c_ephem"] - df["c_base"]) / df["c_base"]
    df["d_refined"] = 100 * (df["c_refined"] - df["c_base"]) / df["c_base"]

    win = df.drop_duplicates(["session", "window", "obj_type", "object", "filter"])
    print("\nrefinement applied (object x filter x window):")
    print(win.groupby(["obj_type", "session", "object"]).agg(
        n=("applied", "size"), applied=("applied", "sum")).to_string())
    a = df[df["applied"] & (df["c_base"] >= 0.05)]
    if not a.empty:
        print("\nwhere applied (raw extremum contrast at the refined position, gate 0.05):")
        print(a.groupby("obj_type").agg(
            n=("applied", "size"),
            offset_px=("offset_px", "median"), offset_sigma=("offset_over_sigma", "median"),
            d_ephem_mean=("d_ephem", "mean"), d_refined_mean=("d_refined", "mean"),
            d_ephem_median=("d_ephem", "median"), d_refined_median=("d_refined", "median"),
            frac_refined_better=("c_refined", lambda s: float((s > a.loc[s.index, "c_ephem"]).mean())),
        ).round(2).to_string())
    na = df[~df["applied"]]
    print(f"\nnot applied: {len(na)} rows, composites identical in "
          f"{int(na['identical_composites'].sum())}/{len(na)}")


if __name__ == "__main__":
    main()
