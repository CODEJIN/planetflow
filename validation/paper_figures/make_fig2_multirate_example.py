"""
Figure 2 of the multi-rate de-rotation paper: intermediate stages of the
multi-rate stacking for one real window (Jupiter, Europa, and its shadow).

Built from the Step 4 outputs written by
    python aperture_contrast.py --tif-dir <session>/TIFs --output-dir <OUT>/<session>
using the same production functions as the pipeline (SatelliteTracker,
_satellite_translate_stack, _capsule_gaussian_mask). Wavelet sharpening is
applied to every panel for display only.

Usage:
    python validation/paper_figures/make_fig2_multirate_example.py \
        --session-dir D:/Astro/real_events_v2/260505 --tif-dir .../260505/TIFs \
        --window window_06 --filter IR --out paper/derotation_flow_gradient_r.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aperture_contrast as ac  # noqa: E402
from pipeline.modules.derotation import find_disk_center, refine_disk_radius_for_plate_scale  # noqa: E402
from pipeline.modules.image_io import read_tif  # noqa: E402
from pipeline.modules.satellite_tracker import SatelliteTracker  # noqa: E402
from pipeline.steps.satellite_composite import (  # noqa: E402
    _apparent_radius_px,
    _capsule_gaussian_mask,
    _satellite_translate_stack,
)

C_SAT = (1.00, 0.62, 0.15)   # satellite: orange
C_SHAD = (0.25, 0.55, 1.00)  # shadow: blue
_COVERAGE_SCALE = 1.5


def _lum(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else img.mean(axis=2).astype(np.float32)


def _display(img: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray:
    sh = ac._apply_wavelet_display(_lum(img), cx, cy, r)
    lo, hi = np.percentile(sh, [0.5, 99.8])
    return np.clip((sh - lo) / max(hi - lo, 1e-6), 0, 1)


def _crop(img: np.ndarray, cx: float, cy: float, r: float, margin: float = 1.12):
    half = int(r * margin)
    y0, x0 = max(0, int(cy) - half), max(0, int(cx) - half)
    return img[y0:int(cy) + half, x0:int(cx) + half], x0, y0


def _overlay(ax, rgb, alpha, x0, y0):
    rgba = np.zeros(alpha.shape + (4,), dtype=np.float32)
    rgba[..., :3] = rgb
    rgba[..., 3] = np.clip(alpha, 0, 1) * 0.75
    ax.imshow(rgba[y0:y0 + ax._crop_h, x0:x0 + ax._crop_w], interpolation="none")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-dir", type=Path, required=True)
    ap.add_argument("--tif-dir", type=Path, required=True)
    ap.add_argument("--window", default="window_06")
    ap.add_argument("--filter", default="IR")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    base_win = args.session_dir / "baseline" / "step04_derotated" / args.window
    comp_win = args.session_dir / "composite" / "step04_derotated" / args.window
    log = json.loads((comp_win / "derotation_log.json").read_text())
    flog = log["filters"][args.filter]
    sat_log, ses_log = log.get("satellite", {}), log.get("session", {})
    t_c = datetime.strptime(log["center_time"], "%Y-%m-%dT%H:%M:%SZ")

    rows = []
    for f in flog["frames"]:
        if f.get("outlier_excluded"):
            continue
        rows.append({"stem": f["stem"], "path": args.tif_dir / f"{f['stem']}.tif",
                     "timestamp": datetime.strptime(f["timestamp"], "%Y-%m-%dT%H:%M:%SZ"),
                     "norm_score": f["norm_score"], "align_shift_px": f.get("align_shift_px", [0.0, 0.0])})
    rows.sort(key=lambda r: r["timestamp"])

    planet = _lum(read_tif(base_win / f"{args.filter}_derotated.tif"))
    final = _lum(read_tif(comp_win / f"{args.filter}_derotated.tif"))
    cx, cy, r, rb, ang = find_disk_center(planet)
    r_ps, _, _ = refine_disk_radius_for_plate_scale(planet, cx, cy, r, rb, ang)

    tracker = SatelliteTracker(flip_ew=bool(ses_log.get("flip_ew", False)),
                               flip_ns=bool(sat_log.get("tracker_flip_ns", ses_log.get("tracker_flip_ns", False))))
    pole_pa = float(sat_log.get("pole_pa_deg", ses_log.get("pole_pa_deg", 0.0)))
    np_ang = float(sat_log.get("np_ang_deg", 0.0))
    ps = tracker.get_plate_scale(r_ps, t_c)
    t_list = [rw["timestamp"] for rw in rows]
    body = tracker.get_positions(t_list, cx, cy, r_ps, pole_pa_deg=pole_pa, np_ang_deg=np_ang)
    shad = tracker.get_shadow_positions(t_list, cx, cy, r_ps, pole_pa_deg=pole_pa, np_ang_deg=np_ang)
    body_ref = tracker.get_positions([t_c], cx, cy, r_ps, pole_pa_deg=pole_pa, np_ang_deg=np_ang)
    shad_ref = tracker.get_shadow_positions([t_c], cx, cy, r_ps, pole_pa_deg=pole_pa, np_ang_deg=np_ang)

    objs = []  # (label, colour, positions, ref, app_r)
    for name, plist in body.items():
        ref = body_ref[name][0]
        if ref is not None and ref.on_disk:
            objs.append((name, C_SAT, plist, ref, _apparent_radius_px(name, t_c, ps)))
    for name, plist in shad.items():
        ref = shad_ref[name][0]
        if ref is not None and ref.on_disk:
            moon = name.replace("_shadow", "")
            objs.append((name.replace("_", " "), C_SHAD, plist, ref, _apparent_radius_px(moon, t_c, ps)))
    if not objs:
        raise SystemExit("no on-disk satellite or shadow in this window")

    stacks, masks = [], []
    for label, col, plist, ref, app_r in objs:
        stacks.append(_satellite_translate_stack(rows, plist, ref))
        traj = [(p.x_px, p.y_px) for p in plist if p is not None and p.on_disk]
        masks.append(_capsule_gaussian_mask(planet.shape, traj, app_r * _COVERAGE_SCALE))

    # ── Figure ───────────────────────────────────────────────────────────────
    raw_idx = [0, len(rows) // 2, len(rows) - 1]
    panels_top = []
    for i in raw_idx:
        img = _lum(read_tif(rows[i]["path"]))
        fcx, fcy, fr, *_ = find_disk_center(img)
        panels_top.append((img, (fcx, fcy, fr), rows[i]["timestamp"], i))

    fig = plt.figure(figsize=(6.7, 7.15), facecolor="black")  # 17 cm printed width
    gs = fig.add_gridspec(3, 6, hspace=0.16, wspace=0.06, left=0.01, right=0.99, top=0.97, bottom=0.01)

    def show(ax, img, geom, title):
        dcx, dcy, dr = geom
        disp, x0, y0 = _crop(_display(img, dcx, dcy, dr), dcx, dcy, dr)
        ax.imshow(disp, cmap="gray", vmin=0, vmax=1, interpolation="none")
        ax._crop_h, ax._crop_w = disp.shape
        ax.set_title(title, color="white", fontsize=7.5, pad=3)
        ax.axis("off")
        return x0, y0

    def mark(ax, x, y, x0, y0, col, size=26):
        ax.scatter([x - x0], [y - y0], s=size, facecolors="none", edgecolors=[col], linewidths=1.2)

    for k, (img, geom, ts, i) in enumerate(panels_top):
        ax = fig.add_subplot(gs[0, 2 * k:2 * k + 2])
        x0, y0 = show(ax, img, geom, f"Input image {i + 1} · {ts:%H:%M:%S} UTC")
        adx, ady = rows[i]["align_shift_px"]
        for label, col, plist, ref, app_r in objs:
            p = plist[i]
            if p is not None:
                # tracker positions are in the aligned frame; undo the frame's alignment shift
                mark(ax, p.x_px - adx, p.y_px - ady, x0, y0, col)

    geom = (cx, cy, r)
    ax = fig.add_subplot(gs[1, 0:2])
    x0, y0 = show(ax, planet, geom, "Stack at planet rate")
    for label, col, plist, ref, app_r in objs:
        for p in plist:
            if p is not None:
                mark(ax, p.x_px, p.y_px, x0, y0, col, size=12)
    for k, (label, col, plist, ref, app_r) in enumerate(objs[:2]):
        ax = fig.add_subplot(gs[1, 2 + 2 * k:4 + 2 * k])
        x0, y0 = show(ax, stacks[k], geom, f"Stack at {label} rate")
        mark(ax, ref.x_px, ref.y_px, x0, y0, col, size=40)

    ax = fig.add_subplot(gs[2, 1:3])
    x0, y0 = show(ax, planet, geom, "Blending bands")
    for (label, col, plist, ref, app_r), m in zip(objs, masks):
        _overlay(ax, col, m, x0, y0)
    ax = fig.add_subplot(gs[2, 3:5])
    show(ax, final, geom, "Final result")

    handles = [plt.Line2D([], [], ls="none", marker="o", mfc="none", mec=col, label=label)
               for label, col, *_ in objs]
    fig.legend(handles=handles, loc="lower left", frameon=False, fontsize=7, labelcolor="white",
               bbox_to_anchor=(0.02, 0.02))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor="black")
    fig.savefig(args.out.with_suffix(".png"), dpi=150, facecolor="black")
    print(f"Saved → {args.out}  ({len(rows)} frames, objects: {[o[0] for o in objs]})")


if __name__ == "__main__":
    main()
