"""
Step 7 – RGB / LRGB compositing (master).

For each time window produced by Step 5/6, builds one composite image per
CompositeSpec defined in config.composite.specs.  Channels are auto-stretched
independently and then aligned to the reference channel before compositing.

Supported composite modes:
  - Plain RGB     (L=None in CompositeSpec)
  - LRGB          (L=filter_name; luminance replaces L in Lab colour space)
  - False colour  (any filter → any channel, no luminance blending)

Default composites (configurable in PipelineConfig):
  • RGB       → R, G, B
  • IR-RGB    → L=IR, R, G, B   (best sharpness: IR carries fine detail)
  • CH4-G-IR  → R=CH4, G=G, B=IR   (methane false colour)

Output (when config.save_step07 is True):
    <output_base>/step07_rgb_composite/
        window_01/
            RGB_composite.png
            IR-RGB_composite.png
            CH4-G-IR_composite.png
            composite_log.json
        window_02/
            …
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pipeline.config import PipelineConfig
from pipeline.modules import composite, image_io


def _color_passthrough(
    config: PipelineConfig,
    results_06: Dict[str, List[Tuple[Optional[Path], str]]],
) -> Dict[str, List[Tuple[Optional[Path], str]]]:
    """Color-camera step 8: pass Step 6 outputs through as 'COLOR' composites.

    No RGB combining needed — the Step 6 output IS already the final color image.
    Copies images into the step07 directory structure so downstream steps (09/11)
    can find them in the usual location.
    """
    out_base: Optional[Path] = None
    if config.save_step07:
        out_base = config.step_dir(7, "rgb_composite")
        out_base.mkdir(parents=True, exist_ok=True)
        print(f"  Output → {out_base}")
    else:
        print("  save_step07=False: color results kept at Step 6 paths")

    all_results: Dict[str, List[Tuple[Optional[Path], str]]] = {}

    for win_label, entries in sorted(results_06.items()):
        # Pick the first available color PNG from Step 6
        src_path: Optional[Path] = None
        for p, _ in entries:
            if p is not None and p.exists():
                src_path = p
                break

        if src_path is None:
            print(f"  [{win_label}] No Step 6 output found — skipped")
            all_results[win_label] = [(None, "COLOR")]
            continue

        out_path: Optional[Path] = src_path   # default: reuse Step 6 path
        if out_base is not None:
            win_out_dir = out_base / win_label
            win_out_dir.mkdir(exist_ok=True)
            out_path = win_out_dir / "COLOR_composite.png"
            shutil.copy2(src_path, out_path)

        print(f"  [{win_label}] COLOR → {out_path.name if out_path else '(not saved)'}")
        all_results[win_label] = [(out_path, "COLOR")]

    return all_results


def run(
    config: PipelineConfig,
    results_06: Dict[str, List[Tuple[Optional[Path], str]]],
) -> Dict[str, List[Tuple[Optional[Path], str]]]:
    """Run Step 7 for all windows produced by Step 6.

    Args:
        config:      Pipeline configuration (composite specs in config.composite).
        results_06:  Output of step06_wavelet_master.run():
                     ``{window_label: [(png_path, filter_name), ...]}``

    Returns:
        ``{window_label: [(composite_path_or_None, composite_name), ...]}``
    """
    # Color camera: no RGB compositing needed — pass through directly
    if config.camera_mode == "color":
        print("  Color camera mode: passing Step 6 color images through as 'COLOR'")
        return _color_passthrough(config, results_06)

    if not results_06:
        print("  [WARNING] No Step 6 results — Step 7 skipped.")
        return {}

    # ── Output directory ───────────────────────────────────────────────────────
    out_base: Optional[Path] = None
    if config.save_step07:
        out_base = config.step_dir(7, "rgb_composite")
        out_base.mkdir(parents=True, exist_ok=True)
        print(f"  Output → {out_base}")
    else:
        print("  save_step07=False: results not written to disk")

    specs = config.composite.specs
    align = config.composite.align_channels
    plow  = config.composite.stretch_plow
    phigh = config.composite.stretch_phigh

    print(f"  Composites: {[s.name for s in specs]}")
    print(f"  Channel alignment: {'enabled' if align else 'disabled'}  "
          f"  Stretch: [{plow}%, {phigh}%]")

    all_results: Dict[str, List[Tuple[Optional[Path], str]]] = {}
    total_written = 0

    for win_label, filter_entries in sorted(results_06.items()):
        # Build {filter_name: png_path} for this window
        filter_paths: Dict[str, Optional[Path]] = {
            filt: path for path, filt in filter_entries
        }

        print(f"\n  {win_label}")

        # Per-window output directory
        win_out_dir: Optional[Path] = None
        if out_base is not None:
            win_out_dir = out_base / win_label
            win_out_dir.mkdir(exist_ok=True)

        win_results: List[Tuple[Optional[Path], str]] = []
        win_log: dict = {"composites": {}}

        for spec in specs:
            # Check required filters are available
            required = {spec.R, spec.G, spec.B}
            if spec.L is not None:
                required.add(spec.L)

            unavailable = {
                f for f in required
                if filter_paths.get(f) is None or not filter_paths[f].exists()
            }
            if unavailable:
                print(f"    [{spec.name}] Missing filters {unavailable} — skipped")
                win_results.append((None, spec.name))
                continue

            # Load filter images
            filter_images = {
                f: image_io.read_png(filter_paths[f])
                for f in required
            }
            # Ensure 2-D (grayscale PNG stored as (H,W) already, but safety check)
            for f in list(filter_images.keys()):
                img = filter_images[f]
                if img.ndim == 3:
                    filter_images[f] = img.mean(axis=2).astype("float32")

            try:
                comp_img, log = composite.compose(
                    spec, filter_images,
                    align=align,
                    max_shift_px=config.composite.max_shift_px,
                    color_stretch_mode=config.composite.color_stretch_mode,
                    stretch_plow=plow,
                    stretch_phigh=phigh,
                )
            except Exception as exc:
                print(f"    [{spec.name}] ERROR: {exc}")
                win_results.append((None, spec.name))
                continue

            out_path: Optional[Path] = None
            if win_out_dir is not None:
                out_path = win_out_dir / f"{spec.name}_composite.png"
                image_io.write_png_color_16bit(comp_img, out_path)
                total_written += 1

            win_log["composites"][spec.name] = log
            win_results.append((out_path, spec.name))
            status = f"→ {out_path.name}" if out_path else "(not saved)"
            print(f"    [{spec.name}] {status}")

        # Save per-window JSON log
        if win_out_dir is not None:
            log_path = win_out_dir / "composite_log.json"
            with open(log_path, "w") as f:
                json.dump(win_log, f, indent=2)

        all_results[win_label] = win_results

    print(f"\n  Step 7 complete: {total_written} composite PNGs written")
    return all_results
