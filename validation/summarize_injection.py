"""
Summarise synthetic_injection.py results into paper-ready tables and a figure.

Usage:
    python validation/summarize_injection.py D:/Astro/synthetic_injection/260417/results.csv [more.csv ...] \
        --out D:/Astro/synthetic_injection/summary
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_COVERAGE_SCALE = 1.5  # capsule sigma_perp = r_app * 1.5 (production value)


def _fmt(s: pd.Series) -> str:
    return f"{s.median():.3f} [{s.quantile(0.25):.3f}, {s.quantile(0.75):.3f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.concat([pd.read_csv(p).assign(source=p.parent.name) for p in args.csv], ignore_index=True)
    base = df[df["delta_deg"] == 0.0]

    # Table A: single-rate vs multi-rate by object type and relative velocity.
    rows = []
    for (obj, dv), g in base.groupby(["obj_type", "dv_R_per_h"]):
        s = g[g["method"] == "single_rate"]
        m = g[g["method"] == "multi_rate"]
        f1 = g[g["method"] == "single_frame_paste"]
        rows.append({
            "obj_type": obj, "dv_R_per_h": dv, "n": len(s),
            "smear_px_median": round(s["baseline_smear_px"].median(), 2),
            **{f"S_{k}": _fmt(s[k]) for k in ["peak_recovery", "flux_recovery", "nrmse", "elongation"]},
            **{f"M_{k}": _fmt(m[k]) for k in ["peak_recovery", "flux_recovery", "nrmse", "elongation"]},
            "M_artifact_rms": _fmt(m["artifact_rms"]),
            **({f"F_{k}": _fmt(f1[k]) for k in ["peak_recovery", "flux_recovery", "nrmse", "elongation"]}
               if len(f1) else {}),
        })
    ta = pd.DataFrame(rows)
    ta.to_csv(args.out / "table_velocity.csv", index=False)

    # Table B: multi-rate sensitivity to angular placement error, paired with the
    # single-rate result of the SAME injected case. The positional offset is
    # expressed in units of the capsule mask width sigma_perp, which is what
    # governs how much of the object the mask actually covers.
    key = ["source", "window", "filter", "obj_type", "dv_R_per_h", "o_ref_x", "o_ref_y"]
    single = df[df["method"] == "single_rate"].set_index(key)["nrmse"].rename("nrmse_single")
    # Rotation about the disk centre is a no-op for an object at the centre.
    mr = df[(df["method"] == "multi_rate") & (df["r_frac"] > 0)].join(single, on=key)
    mr["offset_px"] = mr["r_frac"] * mr["disk_r_px"] * np.radians(mr["delta_deg"])
    mr["offset_over_sigma"] = mr["offset_px"] / (mr["r_app_px"] * _COVERAGE_SCALE)
    mr["better"] = mr["nrmse"] < mr["nrmse_single"]
    rows = []
    for (dv, delta), g in mr.groupby(["dv_R_per_h", "delta_deg"]):
        rows.append({"dv_R_per_h": dv, "delta_deg": delta, "n": len(g),
                     "offset_over_sigma": round(g["offset_over_sigma"].median(), 2),
                     "M_nrmse": round(g["nrmse"].median(), 3),
                     "S_nrmse": round(g["nrmse_single"].median(), 3),
                     "frac_multi_better": round(g["better"].mean(), 2)})
    tb = pd.DataFrame(rows)
    tb.to_csv(args.out / "table_bias.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(ta[["obj_type", "dv_R_per_h", "n", "smear_px_median", "S_peak_recovery", "M_peak_recovery",
                  "S_nrmse", "M_nrmse", "M_flux_recovery", "M_artifact_rms"]
                 + [c for c in ("F_peak_recovery", "F_nrmse") if c in ta.columns]].to_string(index=False))
        print("\nfraction of paired cases where multi-rate beats single-rate (rows: dv, cols: delta deg)")
        print(tb.pivot(index="dv_R_per_h", columns="delta_deg", values="frac_multi_better").to_string())
        print("\nmedian offset / sigma_perp per delta:")
        print(tb.groupby("delta_deg")["offset_over_sigma"].median().to_string())

    # Figure: peak recovery and NRMSE vs relative velocity; NRMSE vs placement offset.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    colors = {"single_rate": "#c0392b", "multi_rate": "#2471a3", "single_frame_paste": "#7f8c8d"}
    labels = {"single_rate": "single-rate", "multi_rate": "multi-rate", "single_frame_paste": "single-frame paste"}
    for ax, metric, ylab in ((axes[0], "peak_recovery", "Peak recovery (1 = truth)"),
                             (axes[1], "nrmse", "NRMSE vs truth (0 = truth)")):
        for obj, ls in (("satellite", "-"), ("shadow", "--")):
            for meth in ("single_rate", "single_frame_paste", "multi_rate"):
                if not (base["method"] == meth).any():
                    continue
                g = base[(base["obj_type"] == obj) & (base["method"] == meth)].groupby("dv_R_per_h")[metric]
                med = g.median()
                ax.plot(med.index, med.values, ls, marker="o", ms=3, color=colors[meth],
                        label=f"{labels[meth]}, {obj}")
                ax.fill_between(med.index, g.quantile(0.25).values, g.quantile(0.75).values,
                                color=colors[meth], alpha=0.12, lw=0)
        ax.set_xlabel(r"Object $-$ surface velocity $\Delta v$ ($R_J$ h$^{-1}$)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    axes[0].axhline(1.0, color="k", lw=0.6)
    axes[0].legend(fontsize=7, frameon=False)

    ax = axes[2]
    shades = {0.1: "#9ecae1", 0.2: "#4292c6", 0.45: "#08306b"}
    for dv, col in shades.items():
        g = mr[np.isclose(mr["dv_R_per_h"], dv)]
        if g.empty:
            continue
        gm = g.groupby("delta_deg").agg(x=("offset_over_sigma", "median"), y=("nrmse", "median"))
        ax.plot(gm["x"], gm["y"], marker="o", ms=3, color=col, label=rf"multi-rate, $\Delta v$={dv:g}")
        ax.axhline(g["nrmse_single"].median(), color=col, ls=":", lw=1)
    ax.plot([], [], ":", color="gray", label=r"single-rate (same $\Delta v$)")
    ax.set_xlabel(r"Placement offset / mask width $\sigma_\perp$")
    ax.set_ylabel("NRMSE vs truth")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(args.out / "injection_summary.pdf")
    fig.savefig(args.out / "injection_summary.png", dpi=150)
    print(f"\nFigure → {args.out / 'injection_summary.pdf'}")


if __name__ == "__main__":
    main()
