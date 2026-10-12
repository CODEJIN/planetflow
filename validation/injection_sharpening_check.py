"""
Does wavelet sharpening amplify the multi-rate gain in the paper's extremum
aperture-contrast metric? Uses the synthetic-injection setup (known truth) and
measures the paper's metric (|P-B|/B, extremum in the ROI) on single-rate vs
multi-rate stacks, both raw and after the same wavelet sharpening used by
aperture_contrast.py.

Usage:
    python validation/injection_sharpening_check.py --tif-dir .../260417/TIFs \
        --out-dir D:/Astro/synthetic_injection/260417 --filter IR
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aperture_contrast as ac  # noqa: E402
import validation.synthetic_injection as si  # noqa: E402
from pipeline.modules import derotation, image_io  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--filter", default="IR")
    ap.add_argument("--n-windows", type=int, default=2)
    ap.add_argument("--dv", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.45])
    args = ap.parse_args()

    res03, cfg = si._run_step03(args.tif_dir, args.out_dir, [args.filter])
    sess = json.loads((args.out_dir / "session_params.json").read_text())
    rows_out = []
    for win in si._pick_windows(res03["windows"], args.n_windows):
        t_ref = win["center_time"]
        B = derotation.query_horizons_sub_observer_lat(horizons_id="599", t_utc=t_ref,
                                                       observer_code=cfg.derotation.observer_code) or 0.0
        sw = {**sess, "B": float(B)}
        rows = win["per_filter"][args.filter]["included"]
        P, _ = si._derotate(rows, t_ref, sw, cfg)
        cx, cy, R, _, _ = derotation.find_disk_center(P)
        centers = {r["stem"]: tuple(derotation.find_disk_center(si._lum(r["path"]))[:2]) for r in rows}
        vx, vy = si._surface_velocity(cx, cy, cx, cy, R, sw, cfg, 1.0)
        n = math.hypot(vx, vy)
        ux, uy = vx / n, vy / n
        for side in (1.0, -1.0):
            o_ref = (side * 0.5 * R * ux, side * 0.5 * R * uy)
            x_ref, y_ref = cx + o_ref[0], cy + o_ref[1]
            for obj, amp in (("satellite", 0.15), ("shadow", 0.85)):
                for dv in args.dv:
                    imgs = {}

                    # Re-run the case, capturing S and M via a thin wrapper.
                    orig_blend = si._blend_poisson

                    def capture(*a, **k):
                        out = orig_blend(*a, **k)
                        imgs.setdefault("M", out)
                        return out

                    si._blend_poisson = capture
                    orig_derot = si._derotate

                    def derot_cap(r_, t_, s_, c_):
                        out = orig_derot(r_, t_, s_, c_)
                        imgs["S"] = out[0]
                        return out

                    si._derotate = derot_cap
                    try:
                        si.run_case(rows, t_ref, P, centers, (cx, cy, R), o_ref, dv, obj, 3.0, amp,
                                    sw, cfg, args.out_dir / "tmp_sharpcheck", 1.0, [0.0])
                    finally:
                        si._blend_poisson = orig_blend
                        si._derotate = orig_derot
                    S, M = imgs["S"], imgs["M"]
                    r_ref = 6.0  # apparent diameter for r_app = 3 px (aperture_contrast.py convention)
                    rec = {"window": f"{t_ref:%H:%M}", "side": side, "obj_type": obj, "dv": dv}
                    for tag, (a, b) in {"raw": (S, M),
                                        "sharp": (ac._apply_wavelet_display(S, cx, cy, R),
                                                  ac._apply_wavelet_display(M, cx, cy, R))}.items():
                        cs = ac.measure_aperture_contrast(a, x_ref, y_ref, r_ref, obj)
                        cm = ac.measure_aperture_contrast(b, x_ref, y_ref, r_ref, obj)
                        rec[f"{tag}_single"], rec[f"{tag}_multi"] = cs, cm
                        rec[f"{tag}_gain_pct"] = 100.0 * (cm - cs) / cs
                    rows_out.append(rec)
                    print(f"{obj:9s} dv={dv:.2f} raw gain={rec['raw_gain_pct']:+6.1f}%  "
                          f"sharp gain={rec['sharp_gain_pct']:+6.1f}%  "
                          f"(sharp single={rec['sharp_single']:.3f} multi={rec['sharp_multi']:.3f})")
    df = pd.DataFrame(rows_out)
    df.to_csv(args.out_dir / "sharpening_check.csv", index=False)
    print(df.groupby(["obj_type", "dv"])[["raw_gain_pct", "sharp_gain_pct"]].median().round(1).to_string())


if __name__ == "__main__":
    main()
