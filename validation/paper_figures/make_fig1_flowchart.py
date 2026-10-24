"""
Figure 1 of the multi-rate de-rotation paper: processing flowchart.

Usage:
    python validation/paper_figures/make_fig1_flowchart.py --out paper/de-rotation_flow.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Fill / edge colours per box family (same scheme as the earlier drawio version).
STYLE = {
    "input":  ("#fff2b3", "#c9a227"),
    "pre":    ("#e6d7ea", "#9673a6"),
    "init":   ("#dae8fc", "#6c8ebf"),
    "ephem":  ("#d5e8d4", "#82b366"),
    "stack":  ("#ffe6cc", "#d79b00"),
    "output": ("#f8cecc", "#b85450"),
}
GROUP_FILL, GROUP_EDGE = "#f5f5f5", "#888888"
TITLE_BAND = 0.048  # vertical space reserved at the top of each group for its title


def box(ax, x, y, w, h, text, kind, fs=8.6, bold=True):
    fc, ec = STYLE[kind]
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.008,rounding_size=0.015",
                                fc=fc, ec=ec, lw=1.1, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=4, linespacing=1.25)


def group(ax, x0, y0, x1, y1, title):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="square,pad=0",
                                fc=GROUP_FILL, ec=GROUP_EDGE, lw=0.9, ls=(0, (4, 3)), zorder=1))
    ax.text((x0 + x1) / 2, y1 - 0.020, title, ha="center", va="center", fontsize=8.8,
            fontweight="bold", color="#333333", zorder=2)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
                                 lw=1.4, color="#222222", zorder=5, shrinkA=0, shrinkB=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    fig = plt.figure(figsize=(6.6, 8.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.015, 1)
    ax.axis("off")

    # Input and lucky stacking
    box(ax, 0.50, 0.965, 0.36, 0.036, "Raw video frames (SER)", "input", fs=9.2)
    arrow(ax, 0.50, 0.946, 0.50, 0.924)
    g_top, g_bot = 0.924, 0.808
    group(ax, 0.20, g_bot, 0.80, g_top, "Lucky imaging")
    box(ax, 0.50, (g_top - TITLE_BAND + g_bot) / 2, 0.52, 0.044,
        "Lucky stacking per filter\n(PlanetFlow or AutoStakkert!4)", "pre")
    arrow(ax, 0.50, g_bot, 0.50, 0.790)

    # Initialization
    g_top, g_bot = 0.790, 0.654
    group(ax, 0.05, g_bot, 0.95, g_top, "Pipeline initialization (from the images)")
    yc = (g_top - TITLE_BAND + g_bot) / 2
    box(ax, 0.285, yc, 0.40, 0.068, "Disk fit, quality scoring\n& de-rotation windows", "init")
    box(ax, 0.715, yc, 0.40, 0.068, "Orientation:\nPole PA, rotation direction,\nN/S (per-window vote)", "init")

    # Branches
    arrow(ax, 0.22, g_bot, 0.22, 0.392)
    arrow(ax, 0.69, g_bot, 0.69, 0.618)

    g_top, g_bot = 0.618, 0.434
    group(ax, 0.42, g_bot, 0.96, g_top, "Ephemeris engine (Skyfield + JPL kernels)")
    box(ax, 0.553, 0.527, 0.225, 0.070, "Satellite\npositions", "ephem")
    box(ax, 0.827, 0.527, 0.225, 0.070, "Shadow positions\n(ray tracing onto\nflattened globe)", "ephem")
    box(ax, 0.69, 0.459, 0.48, 0.026, "Image scale (Minnaert-corrected disk radius)", "ephem", fs=7.8, bold=False)
    arrow(ax, 0.69, g_bot, 0.69, 0.392)

    g_top, g_bot = 0.392, 0.238
    group(ax, 0.04, g_bot, 0.40, g_top, "Planet stack")
    group(ax, 0.44, g_bot, 0.96, g_top, "Satellite / shadow stacks")
    yc = (g_top - TITLE_BAND + g_bot) / 2
    box(ax, 0.22, yc, 0.31, 0.080, "Flattened-globe\nreprojection\n(pre-warp alignment)", "stack")
    box(ax, 0.567, yc, 0.21, 0.080, "Satellite stack\n(shifted along\npredicted path)", "stack")
    box(ax, 0.833, yc, 0.21, 0.080, "Shadow stack\n(shifted along\npredicted path)", "stack")

    # Compositing
    arrow(ax, 0.22, g_bot, 0.38, 0.190)
    arrow(ax, 0.70, g_bot, 0.62, 0.190)
    g_top, g_bot = 0.190, 0.066
    group(ax, 0.14, g_bot, 0.86, g_top, "Compositing")
    yc = (g_top - TITLE_BAND + g_bot) / 2
    box(ax, 0.32, yc, 0.31, 0.050, "Blending band\nalong path", "init")
    box(ax, 0.68, yc, 0.31, 0.050, "Poisson\nblending", "init")
    arrow(ax, 0.50, g_bot, 0.50, 0.050)
    box(ax, 0.50, 0.028, 0.42, 0.040, "Final stack (TIF)", "output", fs=9.4)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(args.out.with_suffix(".png"), dpi=160, bbox_inches="tight", pad_inches=0.03)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
