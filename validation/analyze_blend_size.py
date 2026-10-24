"""
Summarise the blend-method and object-size injection runs.

Reads results_blend_size.csv written by synthetic_injection.py with
--r-app 3 2 5 --r-fracs 0 0.5 0.8 --deltas 0, for one or more sessions, and
prints two comparisons on paired cases (same window, filter, position,
object, size, velocity):

  blend  Poisson (multi_rate) vs Gaussian alpha (multi_rate_alpha) with the
         same stack and capsule mask: NRMSE inside the object aperture,
         trail_rms (deviation from truth along the mask outside the aperture)
         and artifact_rms (change versus the single-rate stack in an annulus).
  size   multi-rate vs single-rate and vs single-frame paste per apparent
         object radius.

Usage:
    python validation/analyze_blend_size.py \
        D:/Astro/synthetic_injection/260417_blendsize/results_blend_size.csv \
        D:/Astro/synthetic_injection/260321_blendsize/results_blend_size.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

KEY = ["session", "window", "filter", "obj_type", "amplitude", "r_app_px",
       "dv_R_per_h", "o_ref_x", "o_ref_y"]


def _load(paths):
    frames = []
    for p in paths:
        d = pd.read_csv(p)
        d["session"] = Path(p).parent.name.split("_")[0]
        frames.append(d[d["delta_deg"] == 0])
    d = pd.concat(frames, ignore_index=True)
    d["r_frac"] = d["r_frac"].round(1)
    return d


def _wide(d, value):
    w = d.pivot_table(index=KEY + ["r_frac"], columns="method", values=value, aggfunc="first")
    return w.reset_index()


def main(paths):
    d = _load(paths)
    print(f"{len(d)} rows, sessions {sorted(d['session'].unique())}, "
          f"r_app {sorted(d['r_app_px'].unique())}, r_frac {sorted(d['r_frac'].unique())}")

    # ── Blend: Poisson vs alpha ─────────────────────────────────────────────
    print("\n== Poisson vs Gaussian alpha (same stack and mask) ==")
    rows = []
    for metric in ("nrmse", "trail_rms", "artifact_rms"):
        w = _wide(d, metric)
        w["ratio"] = w["multi_rate_alpha"] / w["multi_rate"]
        w["poisson_better"] = w["multi_rate"] < w["multi_rate_alpha"]
        for (obj, rf), g in w.groupby(["obj_type", "r_frac"]):
            rows.append({"metric": metric, "obj": obj, "r_frac": rf, "n": len(g),
                         "poisson_med": g["multi_rate"].median(),
                         "alpha_med": g["multi_rate_alpha"].median(),
                         "alpha/poisson": g["ratio"].median(),
                         "poisson_better": g["poisson_better"].mean()})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    w = _wide(d, "nrmse")
    w["ratio"] = w["multi_rate_alpha"] / w["multi_rate"]
    print("\nNRMSE alpha/Poisson by velocity (all positions, r_app 3):")
    g = w[w["r_app_px"] == 3].groupby(["obj_type", "dv_R_per_h"])["ratio"].median().unstack(0)
    print(g.round(3).to_string())

    # ── Size ────────────────────────────────────────────────────────────────
    print("\n== Object size (r_frac 0 and 0.5) ==")
    ds = d[d["r_frac"] <= 0.5]
    w = _wide(ds, "nrmse")
    w["M/S"] = w["multi_rate"] / w["single_rate"]
    w["M/F"] = w["multi_rate"] / w["single_frame_paste"]
    pk = _wide(ds, "peak_recovery")
    w["peak_S"] = pk["single_rate"]
    w["peak_M"] = pk["multi_rate"]
    for obj in sorted(w["obj_type"].unique()):
        print(f"\n{obj}: median per velocity [S nrmse / M nrmse / M/S / M/F / peak S / peak M]")
        g = w[w["obj_type"] == obj].groupby(["r_app_px", "dv_R_per_h"]).agg(
            n=("M/S", "size"), S=("single_rate", "median"), M=("multi_rate", "median"),
            MS=("M/S", "median"), MF=("M/F", "median"),
            pS=("peak_S", "median"), pM=("peak_M", "median"))
        print(g.round(3).to_string())
    print("\nmulti-rate better than single-rate (dv >= 0.1), by size:")
    sub = w[w["dv_R_per_h"] >= 0.1]
    print(sub.assign(b=sub["multi_rate"] < sub["single_rate"]).groupby(["obj_type", "r_app_px"])["b"]
          .agg(["mean", "size"]).round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
