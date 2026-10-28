# PlanetFlow — Algorithm Technical Guide

---

## Table of Contents

1. [Overview](#1-overview)
2. [Step 01 — SER Crop](#2-step-01--ser-crop)
3. [Step 02 — Lucky Stacking](#3-step-02--lucky-stacking)
4. [Step 03 — Quality Assessment & Window Detection](#4-step-03--quality-assessment--window-detection)
5. [Step 04 — De-rotation Stacking](#5-step-04--de-rotation-stacking)
6. [Step 05 / 07 — Wavelet Sharpening](#6-step-05--07--wavelet-sharpening)
7. [Step 06 — RGB Compositing](#7-step-06--rgb-compositing)
8. [Step 08 — Animated GIF](#8-step-08--animated-gif)
9. [Step 09 — Summary Grid](#9-step-09--summary-grid)
10. [Common Module: Disk Detection](#10-common-module-disk-detection-find_disk_center)
11. [Common Module: Sub-pixel Alignment](#11-common-module-sub-pixel-alignment)

---

## 1. Overview

This document explains **what each GUI parameter actually does inside the algorithm**. For usage instructions, see `guide_en.md`. Use this document when you want to understand the meaning of parameters and the mathematical principles behind them.

### Source File Structure

```
pipeline/
├── modules/
│   ├── planet_detect.py        # Step 01: Planet detection and validation
│   ├── lucky_stack.py          # Step 02: Lucky Stacking core algorithm
│   ├── quality.py              # Step 03: Image quality assessment and window enumeration
│   ├── derotation.py           # Step 04: Disk detection, de-rotation warps, stacking
│   ├── satellite_tracker.py    # Step 04: Galilean moon / shadow ephemeris projection
│   ├── position_refinement.py  # Step 04: Opt-in CV position refinement (satellite/shadow)
│   ├── limb_darkening.py       # Minnaert limb-darkening fit (plate-scale radius, opt-in wavelet gain)
│   ├── wavelet.py              # Step 05/07: À trous wavelet sharpening
│   └── composite.py            # Step 06: RGB/LRGB compositing
├── steps/                      # One runner per step (ser_crop.py, lucky_stack.py, …,
│   │                           #   derotate_stack.py, satellite_composite.py, summary_grid.py)
├── data/
│   ├── np_ang_table.json            # Bundled Horizons NP.ang table (offline)
│   └── sub_observer_lat_table.json  # Bundled Horizons sub-observer latitude table (offline)
└── config.py                   # Global configuration (dataclass-based)
```

> **GUI defaults vs. `config.py` defaults**: Some `config.py` dataclass defaults differ from what the GUI actually passes in (e.g. `per_ap_selection`, `use_ncc`, `auto_params`). The tables below list the value **the GUI actually runs with**; where it differs from the dataclass default, this is noted.

---

## 2. Step 01 — SER Crop

**Source**: `pipeline/modules/planet_detect.py`, `pipeline/steps/ser_crop.py`

```
Input frame
    │
    ▼
Convert to 8-bit grayscale
    │
    ▼
GaussianBlur(5×5)  ← Noise suppression
    │
    ▼
Triangle threshold (Zack 1977)
    │
    ▼
Largest connected component (8-connectivity) extraction
    │
    ▼
4-stage validation → reject frame on failure
    │
    ▼
Square ROI crop at bounding box center
```

### GUI Parameters → Internal Behavior

| GUI Parameter | Default | Internal Behavior |
|---|---|---|
| **ROI Size (px)** | 448 | Output square size of `get_cropped_frame()`. Uses `round()` for integer conversion — prevents the 0.5-pixel systematic bias that `int()` would introduce. Pixels outside image boundaries are filled with 0 (black) |
| **Min Diameter (px)** | 50 | Final criterion of the 4-stage validation. Frame is rejected if `max(bw, bh) < min_diameter` |

### Internal Fixed Values

| Parameter | Value | Role |
|---|---|---|
| `padding` | 10 px | Boundary check: planet bounding box must be at least this far from image edges |
| `aspect_ratio_limit` | 0.2 | Aspect ratio check: `min(w,h)/max(w,h) ≥ 1 − 0.2 = 0.8` required to pass |
| `straight_edge_limit` | 0.5 | Straight-edge check: frame rejected if any of the 4 bounding box sides is ≥ 50% lit |
| GaussianBlur kernel | 5×5 | Noise suppression before thresholding |
| `core_percentile` | 60 | Ringed-planet handling (see below) |

### Ringed Planets (Saturn)

Saturn's rings attach to the disk in the thresholded mask and make the bounding box ~2× wider than tall, which would always fail the aspect-ratio check. Before the aspect-ratio / straight-edge checks run, the blob is re-thresholded at its own `core_percentile` (60th) brightness percentile to isolate the brighter disk core, and those two checks are applied to the core only. Boundary, minimum-diameter and centre still use the full disk+ring extent.

Some filters invert this contrast (e.g. Saturn's CH4 band, where the globe is darker than the rings). Instead of special-casing a filter name, a failure to isolate a plausible round core is treated as "brightness cannot validate this frame's shape", and the aspect/edge checks are skipped for that frame.

### Triangle Auto-threshold

Implemented with OpenCV's `THRESH_TRIANGLE` flag. Determines the threshold by finding the minimum point most distant from the histogram's highest peak. Operates stably across a wide exposure range regardless of aperture settings or planet size.

### Bounding Box Center Crop

Jupiter has non-uniform brightness due to belts and the Great Red Spot. A brightness-weighted centroid would introduce systematic bias toward bright structures. To prevent centering bias, **bounding box center** `(x + w/2, y + h/2)` is used.

---

## 3. Step 02 — Lucky Stacking

**Source**: `pipeline/modules/lucky_stack.py`, `pipeline/steps/lucky_stack.py`

The GUI always runs Lucky Stacking with `per_ap_selection=True` and `use_ncc=True`. That routes stacking to **per-AP independent lucky stacking** (`_per_ap_independent_stack()`): each alignment point (AP) picks its own best frames by *local* sharpness, so different regions of the disk can be built from different frame subsets.

```
SER input file
    │
    ▼
[1] Frame quality scoring (log_disk metric, every 2nd frame + interpolation)
    │
    ▼
Select top top_percent% frames → selected_indices
    │
    ▼
[2] Reference frame construction
    reference_n_frames (50) frames centred on the 75th-percentile quality rank
    → phase-correlation alignment to the first of them → mean stack
    → find_disk_center() → (cx, cy, disk_radius)
    → Saturn only: fit disk+ring blob ellipse (ring-aware AP extent)
    │
    ▼
[3] AP grid generation (uniform grid, or AS!4 greedy PDS 3 layers)
    │
    ▼
[4] Pass 1 — for every selected frame:
    global alignment: limb_center_align() (fallback: phase correlation
    if the shift exceeds 0.5 × disk_radius) → bicubic sub-pixel shift
    per-AP local score: LoG variance inside each AP patch
    │
    ▼
[5] Pass 2 — for every AP independently:
    pick the top per_ap_stack_sub_percent (50%) frames by THIS AP's score
    per-frame local shift: Hann-windowed NCC + QSF sub-pixel peak
    extract a 2 × ap_size patch at the shifted position
    weighted mean (weight = local_score ^ quality_weight_power)
    │
    ▼
[6] Scatter-blend all AP patches with a Gaussian mask (σ = 2·ap_size / 3)
    → normalise by accumulated mask weight
    │
    ▼
[7] Gaussian rolloff in the frequency domain (σ_f = 0.20)
    │
    ▼
If n_iterations = 2: use the result as the reference → repeat [4]–[7]
    │
    ▼
Output TIF
```

### GUI Parameters → Internal Behavior

| GUI Parameter | Default | Internal Behavior |
|---|---|---|
| **Top Frame Percent (%)** | 25 | `top_percent = 0.25`. Only the top N% of frames by quality score enter stacking. `n_select = max(min_frames, round(n_frames × top_percent))` |
| **AP Size (px)** | 64 | Base AP size s. Uniform grid: spacing = s ÷ 2. AS!4 PDS: layers s, round(s×1.5/8)×8, s×3. Each AP blends a 2s × 2s patch. |
| **N Iterations** | 1 | `n_iterations`. With 2, the 1st-pass stack becomes the reference for the 2nd pass → higher-SNR reference for the per-AP NCC shifts, at ~2× the time. In practice the gain over 1 pass is negligible. |
| **SER Parallel** | 1 | Number of SER files processed at once. 0 = auto (CPU cores ÷ 4). Each SER gets `n_workers ÷ N_SER` threads. ~950 MB RAM per SER. |
| **AS!4 AP Grid** | Off | Off = uniform grid. On = AS!4 greedy Poisson-disk sampling, generated **once per session** from a reference SER and shared across all SERs (shifted by each SER's own disk offset). |
| **Debayer** | On | Color camera only. Bayer frames are debayered to RGB **before** the sub-pixel shift (shifting a Bayer mosaic would mix colour sites). NCC shift estimation still runs on the Bayer luminance. |

### Internal Fixed Values

| Parameter | Value | Role |
|---|---|---|
| `score_metric` | `"log_disk"` | Frame quality metric (matches AS!4 *lapl3*). Also available in config: `"local_gradient"`, `"laplacian"`, `"gradient"` |
| `reference_midpoint_percentage` | 75 | Reference window centred on the 75th-percentile rank ("solidly good" frames rather than rare lucky outliers — AS!4 default) |
| `reference_n_frames` | 50 | Frames mean-stacked into the reference |
| `per_ap_stack_sub_percent` | 0.5 | Fraction of the selected pool each AP stacks (0.5 × 25% ≈ top 12.5% globally, per AP) |
| `quality_weight_power` | 3.0 | Per-frame patch weight = local_score³ |
| `ap_search_range` | 20 px | Per-AP local shifts larger than this are discarded (shift → 0) |
| `ap_min_brightness` | 0.196 | AP rejected if its patch mean is below this (AS!4 Min Bright 50/255) |
| `ap_min_contrast` | 0.01 | Uniform grid only: AP rejected if patch RMS contrast is below this |
| `fourier_rolloff_sigma` | 0.20 | Output low-pass sigma in normalised frequency (0 = DC, 0.5 = Nyquist) |
| `remap_interpolation` | `INTER_CUBIC` | Interpolation for sub-pixel shifts |
| `sigma_clip` | Off | Optional post-pass that re-warps all frames against the final stack and rejects pixels beyond `sigma_clip_kappa` (2.0) σ. Config/session only — no GUI control. |

### Frame Quality Scoring (`log_disk`)

Matches AS!4's *lapl3*. Spearman correlation 0.74 vs. AS!4 frame rankings.
```
mask  = (frame / max) > 0.25
score = var(Laplacian(GaussianBlur(frame, σ=3.0)))  on mask
```
The same LoG-variance measure is evaluated per AP patch in Pass 1 to rank frames locally.

### AP Grid Placement

**Uniform grid** (AS!4 AP Grid Off): AP centres every `ap_size ÷ 2` px. An AP is kept if its centre is inside the disk, its patch fits inside the image, and it passes the brightness and contrast floors.

**Greedy PDS** (AS!4 AP Grid On): 3 independent layers, generated by raster scan:

| Layer | AP size (s=64) | Minimum AP spacing |
|-------|----------------|--------------------|
| Layer 1 | 64px | `round(64 × 35/64)` = 35px |
| Layer 2 | 96px (`round(64×1.5/8)×8`) | `round(96 × 35/64)` = 52px |
| Layer 3 | 192px (`64×3`) | `round(192 × 35/64)` = 105px |

Acceptance per AP: ① inside the disk, ② mean patch brightness ≥ 0.196 (integral image, O(1)), ③ minimum spacing to already-accepted APs of the same layer. Match rate vs. AS!4 ground truth: 96–100%.

### Ring-aware AP Extent (Saturn)

For Saturn (`horizons_id == "699"`) this is applied automatically; there is no toggle.

- **Eligible region**: an ellipse is fitted to the whole illuminated blob (disk + rings): Otsu − 10%, 7×7 closing, largest contour, `cv2.fitEllipse`. APs may sit anywhere inside that ellipse instead of only inside the disk radius. `find_disk_center()`'s ring-stripped disk radius is still used everywhere else.
- **AS!4 PDS**: each layer first runs its disk-only pass **unchanged**, then a second pass adds ring-only candidates against the same spacing grid. This keeps the validated AS!4 disk placement intact. (An earlier single-pass version moved up to ~99% of disk APs because near-limb candidates won the raster race.)
- **Per-AP blending**: a ring AP's 2s × 2s blend mask can reach deep into the disk. The mask is therefore zeroed (with a 3 px feather) wherever it falls inside the disk radius, so ring content never blends into disk pixels. Disk APs are unchanged.
- **Session-wide APs**: the ring ellipse is passed along with the session APs, so per-SER re-filtering does not drop the ring APs.
- **Preview**: the Step 02 preview draws the ring ellipse in magenta and labels the grid `+Ring`.

### Alternative Stacking Paths (config.py only)

These are reachable only by editing `config.py` (with `per_ap_selection=False`); the GUI never selects them. Their parameters (`use_tps`, `fourier_quality_power`) are therefore config-only; the Step 02 panel no longer shows Warp Method or Fourier Quality Power.

- **Fourier-domain quality-weighted stacking** (`use_fourier_quality=True`, Mackay 2013, arXiv:1303.5108): each globally aligned frame contributes `|F_n(f)|^power × F_n(f)` per spatial frequency, normalised by `Σ|F_n(f)|^power`, then the 0.20 rolloff. Only global alignment is applied (no per-AP warp).
- **Warp-field stacking**: per-AP Hann-windowed shifts (phase correlation or NCC + QSF) are interpolated into a full-resolution warp field by Gaussian kernel regression, `σ = ap_step × ap_sigma_factor` (0.9), or by TPS. KR produces a C∞-continuous field, unlike Delaunay's C⁰ mesh, whose triangle-edge creases accumulate over thousands of frames and become visible after wavelet sharpening. For Saturn, disk and ring APs are regressed as two independent fields and composited with a feathered seam at the disk boundary.

**Parallelism model** (e.g., 32 cores, 4 SER parallel):
```
Total thread budget = n_workers (default: all cores)
SER-level:   4 SER files processed simultaneously
Frame-level: each SER uses n_workers ÷ 4 = 8 threads
```

---

## 4. Step 03 — Quality Assessment & Window Detection

**Source**: `pipeline/modules/quality.py`, `pipeline/steps/quality_assess.py`

```
Step 02 TIF file list
    │
    ▼
Per TIF image:
    Otsu threshold (−15% margin) → largest component → disk mask
    GaussianBlur(σ=1.2) → denoise
    Laplacian variance (×0.5) + Tenengrad (×0.3) + Normalized variance (×0.2)
    → composite raw_score
    │
    ▼
Per-filter min-max normalization → norm_score ∈ [0, 1]
    │
    ▼
Drop frames with norm_score < Min Quality Threshold
    │
    ▼
One candidate window per anchor frame (IR if present, else the first filter):
    window = anchor time ± window_minutes / 2
    Per filter: low-tail σ-clipping (mean − 1.5σ) → quality_post × snr_factor × stability
    │
    ▼
Geometric mean across filters → window_quality
    │
    ▼
All windows, sorted chronologically (overlaps included)
    │
    ▼
Output: windows.json / *_ranking.csv
```

### GUI Parameters → Internal Behavior

| GUI Parameter | Default | Internal Behavior |
|---|---|---|
| **Window (frames)** | 3 | Window length in filter cycles. `window_minutes = frames × cycle_minutes`, centred on each anchor frame |
| **Cycle Seconds** | 225 | Duration of one filter cycle (IR→R→G→B→CH4→IR). Sets the window length and the expected frame count `n_expected = window_minutes / cycle_minutes` used by `snr_factor`. In color camera mode the label becomes "single frame interval" with a 45 s default |
| **Min Quality Threshold** | 0.05 | Frames with `norm_score < threshold` are removed **before** the window search. 0.0 includes all frames |

### Internal Fixed Values

| Parameter | Value | Role |
|---|---|---|
| Laplacian weight | 0.5 | Fraction of Laplacian variance in the composite score |
| Tenengrad weight | 0.3 | Fraction of Tenengrad (mean squared Sobel) in the composite score |
| Normalized variance weight | 0.2 | Fraction of `var/mean` in the composite score |
| Denoise σ | 1.2 px | Gaussian blur before sharpness metrics. Prevents noisy-but-blurry frames from scoring high |
| σ-clipping threshold | 1.5σ, low side only | A frame is excluded if `norm_score < mean − 1.5σ` within its window; unusually good frames are never clipped |

### Window Quality Calculation

```
# Per filter
quality_post = mean(norm_score of included)
snr_factor   = min(1.0, √(n_included / n_expected))
stability    = 1 / (1 + CV)          CV = std/mean of included
filter_quality = quality_post × snr_factor × stability

# Across all filters
window_quality = (∏_f  filter_quality_f) ^ (1 / num_filters)
```

**Why geometric mean**: if any one filter is very poor, the window quality drops substantially. All filters must meet a minimum standard to produce a good composite.

A window is dropped if any required filter has no frame in it. `rotation_degrees` in `windows.json` is a rough estimate at Jupiter's ~0.6°/min and is for display only; Step 04 computes the actual rotation from timestamps and the configured period.

**Window selection is not done here**: Step 03 returns every window so that Steps 04–06 can process the whole session. Top-N and non-overlap selection happen in Step 09.

---

## 5. Step 04 — De-rotation Stacking

**Source**: `pipeline/steps/derotate_stack.py`, `pipeline/modules/derotation.py`, `pipeline/steps/satellite_composite.py`

```
Step 02 TIF + Step 03 windows
    │
    ▼
Session-level (once):
    ├─ pole_pa: image-space belt-axis angle, accumulated over all frames
    │   of the preferred filter (IR > R > G > B > CH4)
    ├─ derot_flip: warp direction, from forward-prediction NCC on real drift
    ├─ flip_pole_axis: 3D reprojection's sign ambiguity, same technique
    ├─ tracker_flip_ns: camera N/S orientation (satellite tracker only)
    └─ derot_confidence: high-pass NCC sweep (diagnostic only)
    │
    ▼
Per window:
    ├─ NP.ang (tracker only) + sub-observer latitude B (bundled table → cache → Horizons)
    └─ Per filter:
        ├─ Reference frame = frame closest to the window centre time
        ├─ Disk detection on the reference; cross-filter reconciliation
        │   (shared radius / shape / pose — see below)
        ├─ Per frame:
        │   ├─ Pre-warp disk centre + scale → applied to the RAW frame
        │   ├─ True 3D oblate-spheroid reprojection warp (Δt from window centre)
        │   └─ Saturn: ring occlusion weight suppresses atmosphere warp on ring pixels
        ├─ Alignment fallback chain + outlier-shift rejection
        └─ Quality-weighted mean (weights = norm_score ^ stack_weight_power)
    │
    ▼
pole_pa re-measured on the de-rotated output (for the tracker)
    │
    ▼
Optional: satellite/shadow composite → overwrites the per-filter TIFs
```

### GUI Parameters → Internal Behavior

| GUI Parameter | Default | Internal Behavior |
|---|---|---|
| **Min Quality Threshold** | 0.05 | Frames with `norm_score < threshold` are dropped before stacking. If all frames would be dropped, the filter proceeds with all of them. |
| **Stack Weight Power** | 1.0 | `weights = norm_score ^ power`. 1.0 = linear blend. 2.0–4.0 favours the best frames when quality varies a lot within one window (validated on real Saturn data; not a universal default). |
| **Normalize Brightness** | Off | Rescales each frame so its disk median matches the reference frame. Discards real brightness information — use only when blotchy artifacts are visible. |
| **Satellite Composite** | Off | Enables the satellite tracker **and** compositing of every on-disk Galilean moon and shadow (see below). |
| **CV Position Refinement** | Off | Can only be enabled when Satellite Composite is on. Per-window, confidence-gated CV correction of the ephemeris-predicted positions (see below). The status label next to it shows whether CH4 data is available. |

**Set in Global Settings** (from the planet preset): rotation period, **Warp Scale** (default 1.0 for every planet), and **True Polar/Equatorial Ratio** (Jupiter 0.9351, Saturn 0.9021, Mars 0.9941, Uranus 0.9771, Neptune 0.9829).

### Internal Fixed Values

| Parameter | Value | Role |
|---|---|---|
| `use_true_reprojection` | always On | The GUI forces the true 3D reprojection warp; the linear warp is reachable only from code |
| `req_px` (R) | `disk_radius × 1.05` | 5% padding of the warp sphere, avoiding the limb singularity |
| `_interp_feather_px` | 12.0 px | CUBIC (interior) / LINEAR (limb) interpolation blend, cosine fade over 12 px inside the limb |
| `_RADIUS_SHARE_REL_TOL` | 8% | Per-filter radius snaps to the window median only if already within 8% |
| `_SCALE_MIN` / `_SCALE_MAX` | 0.95 / 1.05 | Clamp for the per-frame scale correction |
| `_OUTLIER_ABS_PX` | 5.0 px | Hard floor for outlier-shift rejection |

### pole_pa — Measured from the Image, Not from Horizons

`pole_pa_deg`, which sets the warp's drift direction, is an **image-space** angle. `auto_detect_equator_pa()` builds a gradient-angle histogram over the inner disk (r < 0.75R) from two wavelet-sharpened views: fine 2–8 px belt edges and 8–32 px belt widths. The dominant belt-edge gradient direction is the projected pole axis; the result is rotated 90° to give the drift axis. Coarser wavelet levels are excluded because they pick up the radial limb-darkening gradient and bias the angle by ~10°.

The session value accumulates **one histogram over every frame** of the preferred filter before taking the argmax. Frames where the belt-gradient path fails fall back to `equator_pa_from_disk_ellipse()`.

**NP.ang** (Horizons quantity 17, the celestial north-pole angle) is **not** used for the warp. It is used only by the satellite tracker, as camera rotation `θ_cam = pole_pa + NP.ang`. The two angles live in different frames and are not interchangeable.

After stacking, `pole_pa` is **re-measured on the de-rotated output** and that value is given to the tracker. The tracker draws on the de-rotated image, whose effective orientation was measured to differ by ~0–3.5° from the pre-warp value.

### Warp Direction Auto-detection

- **`derot_flip`** (`auto_detect_ns_flip()`): warps an early frame forward with both drift directions and keeps the one whose prediction better matches a later frame (NCC on the inner disk). For both N-up and pure N/S-flipped cameras, features drift the same way in the image, so this normally returns `False`. It does **not** tell the satellite tracker which way is north.
- **`flip_pole_axis`** (`auto_detect_pole_axis_flip()`): the same forward-prediction test applied to the 3D reprojection's own sign ambiguity, which exists only once sub-observer latitude B is modelled. Ambiguous results (|Δcorr| < 0.001) default to `False`.

Neither is a GUI setting; both are detected once per session.

### True 3D Oblate-spheroid Reprojection Warp

**Source**: `spherical_derotation_warp_3d()`

For every output pixel, the warp:

1. **Unprojects** the pixel onto an oblate spheroid (equatorial radius `req_px`, polar radius `req_px × true_polar_equatorial_ratio`), tilted by the sub-observer latitude B and rotated to `pole_pa`, to get its planetocentric latitude/longitude.
2. **Shifts longitude** by `Δλ = (Δt / period) × 2π`, with Δt measured from the window centre time.
3. **Reprojects** the point orthographically and samples the source frame there (CUBIC inside the disk, LINEAR near the limb, 12 px cosine blend).

Unlike the older linear warp (`drift = scale × Δλ × depth(x, y)` along one axis), this models B explicitly. It must therefore use the planet's **true** physical oblateness, not the fitted ellipse's apparent aspect ratio, which already contains B-foreshortening. The rotation is geometrically exact, so `warp_scale` should stay at 1.0; any other value triggers a runtime warning that something else (timestamps, period, orientation) is wrong.

**Saturn rings** (`has_rings`, `horizons_id == "699"`): the rings do not co-rotate with the atmosphere. `compute_ring_occlusion_weight_3d()` builds a continuous [0, 1] weight from Saturn's physical ring radii (C-ring inner edge 1.239 Req, A-ring outer edge 2.269 Req), B and `pole_pa`, marking where **foreground** ring material covers the globe. There, the warped pixel value is blended back toward the frame's own un-warped content, so atmosphere rotation is never applied to ring pixels.

### Cross-filter Disk Geometry Reconciliation

Each filter detects its disk on its own reference frame. Three reconciliations then apply, each only where needed:

- **Shared radius** (`resolve_shared_radius()`): within a window, each filter's semi-major axis differs by 1–3 px from Otsu noise, which would cut each filter's warp off at a slightly different radius and show up as a limb colour fringe. A filter snaps to the window median radius **only if** it is already within 8% of it; a genuinely different size (e.g. a CH4 absorption band) is left alone.
- **Shared shape** (`resolve_shared_shape()`): a filter whose own fit cannot measure oblateness/orientation reliably (e.g. Saturn CH4, where the disk is darker than the rings) borrows the aspect ratio and angle from a sibling filter.
- **Filter pose** (`resolve_filter_pose()`): only when a filter's own detection fails outright, its centre is registered against a sibling filter's frame.

### Pre-warp Alignment and Scale Correction

Seeing makes the disk wobble ~0–2 px from frame to frame. For each frame, `find_disk_center()` runs on the **raw** (pre-warp) luminance image:

- Translation: `(ref_cx − cx_i, ref_cy − cy_i)`, accepted if both components are within 15 px.
- Scale: `own_ref_radius / radius_i`, accepted only for a confident fit and within [0.95, 1.05].

The correction is applied to the **raw frame before warping** (`apply_shift_and_scale()`). The warp is a non-rigid, depth-dependent deformation defined around the assumed centre, so warping first and shifting afterwards are not equivalent. Applying the shift after the warp left a median 0.58 px (up to 1.38 px) residual on real Saturn data.

Post-warp centre detection is not used as a general correction. The warp moves belts near the limb, which drags the detected centre in the warp direction and partially undoes the de-rotation.

**Fallback chain** (logged in `derotation_log.json` as `align_method`):

| `align_method` | Source | Condition |
|---|---|---|
| `"reference"` | No shift | The reference frame itself |
| `"pre_warp_center"` | Pre-warp disk centre (+ scale) | Normal case |
| `"limb_center"` | Post-warp limb centre | Pre-warp detection failed |
| `"phase_correlate"` | Phase correlation | `limb_center` returned zero shift |

For a filter whose own shape detection is unreliable (e.g. Saturn CH4), the pre-warp shift comes from phase correlation over the disk region instead, with no scale correction.

**Outlier-shift rejection**: a non-reference frame is dropped if its alignment shift is more than `max(1.5, 4 × robust σ)` **and** more than 5 px from the window median. This catches silent registration failures, such as a CH4 frame needing ~8 px while its window-mates need ~1 px, which would otherwise smear detail.

### De-rotation Confidence Measurement (NCC Sweep)

**Source**: `pipeline/steps/derotate_stack.py` → `_measure_derot_confidence()`

A **diagnostic** that answers "given the configured warp_scale, how well does the belt structure support the de-rotation?". It never changes `warp_scale`.

1. Take the session's longest window (minimum 3° of rotation) and its earliest and latest frames.
2. Sweep `scale` from 0.50 to 1.20 in 13 steps, plus the configured value.
3. Warp the early frame forward by the full Δt and compute high-pass NCC against the late frame on the inner disk (r ≤ 0.7 × semi_a).
4. Report the NCC at the configured scale as `derot_confidence`.

The high-pass filter subtracts a σ = 30 px Gaussian blur first. Without it, the smooth limb-darkening gradient dominates the NCC and the smallest scale always wins.

The estimated peak scale is logged as `est. peak` for diagnostics only. On real data the NCC-vs-scale curve was found to be noise-dominated with no genuine peak, so it must not be used to pick `warp_scale`.

**Warning threshold**: if the NCC is below 0.80, a GUI warning appears after Step 04 (standalone) or after Run All, recommending a shorter Step 03 window.

### Horizons Lookups

`query_horizons_np_ang()` and `query_horizons_sub_observer_lat()` use the same priority:

1. **Bundled tables** (offline): `pipeline/data/np_ang_table.json` and `sub_observer_lat_table.json`, covering Jupiter (599), Saturn (699) and Mars (499) for 2016–2036, linearly interpolated (NP.ang with 360°/0° wrap handling).
2. **User cache**: `~/.astropipe/horizons_cache.json`.
3. **Live Horizons API**: outside the bundled range, or for Custom planets.

Sub-observer latitude is queried only when the 3D reprojection or ring handling needs it (always, given the GUI forces 3D reprojection). If unavailable, 0.0° is used with a warning.

### Satellite / Shadow Composite

**Source**: `pipeline/steps/satellite_composite.py` → `_apply_satellite_composite()`

With **Satellite Composite** on, every Galilean moon (Io, Europa, Ganymede, Callisto) and every moon shadow predicted to be **on the disk** at the window centre is composited into each filter's de-rotated TIF, using a capsule-shaped mask and Poisson gradient-domain blending. Off-disk moons are skipped.

```
For each filter TIF:
    Detect this filter's disk centre (disk_cx, disk_cy)
        │
        ▼
    Query canonical moon + shadow positions at t_center (Skyfield)
    in THIS filter's disk coordinate system
        │
        ▼
    (Optional) CV position refinement
        │
        ▼
    Query per-frame positions; translate-stack raw frames so the moon/shadow
    is aligned at its canonical position
        │
        ▼
    Solve the Poisson equation inside the capsule mask → splice into the planet
        │
        ▼
    Overwrite the filter TIF in place
```

#### Per-filter Disk Coordinate System

Each filter's canonical position is computed in **that filter's own disk coordinate system**. Each filter's de-rotated disk can sit at a slightly different absolute pixel position, and Step 06's `align_channels()` later shifts channels to align their disks. Because the moon's **disk-relative** offset is identical in every filter, Step 06's alignment moves disk and moon together, and the moon lands in the same place in every composite.

#### Blending — Capsule Mask + Poisson

```
alpha(x,y) = exp(−min_dist_to_trajectory(x,y)² / (2 × sigma_perp²))
sigma_perp  = apparent_radius_px × coverage_scale     (coverage_scale = 1.5)

∇²result = ∇²sat_stack    (inside mask)
result    = planet         (mask boundary — Dirichlet BC)
```

The mask width follows the moon's apparent size and its length follows the actual motion path, so it stays tight during long transits. Only the gradient structure is transferred; the DC level comes from the planet boundary, so per-filter DC residuals cannot cause a colour cast.

#### plate_scale

`plate_scale = ang_radius_geometric / r_ref`, where `r_ref` is the session-median disk radius. Each frame's raw `find_disk_center()` radius is first corrected by `refine_disk_radius_for_plate_scale()`. The Otsu threshold crosses the limb-darkening falloff well inside the true geometric edge, undersizing the radius by ~6–18%. A Minnaert-law fit (`limb_darkening.py`) to the disk's own radial profile recovers the geometric radius. It falls back to the raw radius on fit failure or a ratio outside [1.0, 1.3]. The corrected radius is used **only** for plate scale and positions; warp and feathering keep the original radius.

Moon/shadow pixel positions are the pure output of the ephemeris projection. CV-fitted global calibration of plate scale or disk centre was removed on 2026-08-31: image-based "ground truth" on Jupiter was shown to be fooled by belt structure into confident false detections.

#### Shadow Position Algorithm (Oblate-spheroid Ray Intersection)

**Source**: `pipeline/modules/satellite_tracker.py` → `_shadow_pos_skyfield()`

1. **Light time**: geometry is evaluated at `t_emit = t_obs − d_EJ/c` (≈ 47 min for Jupiter). The Sun endpoint is back-dated again by its own Sun→Jupiter light time (~43 min).
2. **Ray-spheroid intersection**: a ray from the Sun through the moon is intersected with Jupiter as an oblate spheroid (R_eq = 71 492 km, R_pol = 66 854 km, IAU pole). The near-hemisphere hit is the shadow. Valid only when λ > 1 (Sun → moon → surface).
3. **Projection**: the point is converted to apparent RA/Dec from Earth at `t_obs`, including **stellar aberration** from Earth's velocity, then projected to pixels with `plate_scale`, `θ_cam = pole_pa + NP.ang` and the flip convention.

> **Known limitation, unresolved**: a systematic *rotational* offset of ~5–14° (average ~+7–8°) remains between predicted and actual shadow positions. Many candidate causes were tested and ruled out. One partial cause (~1–3°), measuring `pole_pa` on the raw frame instead of the de-rotated output, was fixed. At typical transit radii the residual is ~9 px, comparable to the object's own radius. See [`investigation_shadow_rotational_bias.md`](investigation_shadow_rotational_bias.md).

#### CV Position Refinement (opt-in)

**Source**: `pipeline/modules/position_refinement.py`

A narrowly scoped, **per-window** correction, unlike the rejected global calibration above:

- **Moon body**: only CH4 shows reliable contrast (methane darkens the disk, not the moons). The offset is measured once per window on CH4 and added to every filter's own predicted body position. It is skipped silently if the session has no CH4.
- **Shadow**: dark in every filter, so each filter refines its own shadow independently.
- **Gating**: re-detection runs at four box sizes (2–5× the object radius). An offset is applied only if detection clears a **3σ** confidence bar (stability across box sizes + peak significance) **and** the offset is ≤ 20 px. Otherwise pure ephemeris is used. Nothing is cached across windows or sessions.

#### Camera N/S Orientation (tracker_flip_ns)

**Source**: `satellite_composite.py` → `resolve_tracker_flip_ns()`, `satellite_tracker.py` → `detect_tracker_flip_ns()`

The tracker must know whether north is up or down in the image; `derot_flip` cannot answer this. Priority:

1. An explicit `satellite.flip_ns` in config.
2. **Belt-asymmetry auto-detection, voted per window**: each window's own frames are averaged and tested separately. The SEB is wider and darker than the NEB; `score = 0.7 × width_asymmetry + 0.3 × brightness_signal`, `confidence = min(1, 2.5 × |score|)`. Windows with confidence ≥ 0.25 vote; the majority wins. Averaging the whole session at once was abandoned: Jupiter rotates ~36°/h, so hours of frames smear the belts (one real session gave 0.02 session-wide vs. 0.58–0.80 per window).
3. If neither resolves, satellite/shadow tracking is **skipped** for the session. Falling back to `derot_flip` produced errors up to ~60 px on real data.

#### Shadow Ephemeris Files (Skyfield BSP)

| File | Size | URL |
|---|---|---|
| `de440s.bsp` | 32 MB | `naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/` |
| `jup365.bsp` | 1.1 GB | `naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/` |

Storage resolution order: `PLANETFLOW_SKYFIELD_DIR` → `~/.planetflow/skyfield/` → `/tmp/skyfield/`. Missing files are downloaded automatically on first run when internet is available.

The **BSP status indicator** next to the checkbox runs a background check:
1. `skyfield` not importable → red, checkbox disabled (`pip install skyfield` required)
2. BSP files present → green (OK)
3. Files missing, `naif.jpl.nasa.gov:443` reachable → orange (auto-download on first run); otherwise red, checkbox disabled

### Opt-in Experimental Options (config.py only)

These are off by default, have no GUI control, and are documented with their validation status in `DerotationConfig`:

| Option | Status |
|---|---|
| `sharpness_selection_enabled` | **Do not enable**: introduces a thin bright arc inside the limb on Jupiter and Saturn |
| `s0_sl_blend_enabled` | Blends the stack toward the reference frame where rotation coverage is poor; unvalidated |
| `compute_ring_only_stack` | Saturn ring-only companion stack (`<filt>_ring_only.tif`); measured slightly worse than the normal stack, not recommended |

---

## 6. Step 05 / 07 — Wavelet Sharpening

**Source**: `pipeline/modules/wavelet.py`, `pipeline/steps/wavelet_master.py`, `pipeline/steps/wavelet_preview.py`

Both steps share the same à trous decomposition and gain model, but differ in masking:

| | Step 05 (Master) | Step 07 (Preview) |
|---|---|---|
| Input | Step 04 de-rotated TIFs | Step 02 TIFs |
| Function | `sharpen_disk_aware()` — gain feathered to 0 at the limb | `sharpen()` / `sharpen_color()` — uniform gain, no disk mask |
| Auto edge parameters | Always on (`auto_params=True` from the GUI) | — |
| `sharpen_filter` (noise gate) | 0.0 (off) | 0.1 |
| Saturn ring mask | Yes (see below) | — |

Step 07 deliberately skips the disk-aware mask: a circular feather cuts into Jupiter's oblate equatorial bulge, and a lucky-stack TIF already has a smooth limb fade, so plain sharpening does not ring at the limb.

### Step 05 Pipeline

```
Step 04 TIF
    │
    ▼
Disk detection → (cx, cy, rx, ry, angle)
    │
    ▼
auto_wavelet_params():
    expand_px = sqrt(rx·ry) × 0.0505        (push the mask edge out to the visual limb)
    eff       = limb-inward brightness-gradient width / 2
                (fallback: sqrt(rx·ry) × 0.030)
    │
    ▼
À trous B3 decomposition (6 levels) → [detail_0 (~2px), …, detail_5 (~64px), residual]
    │
    ▼
Per level i:
    detail_i → per-layer denoise (soft threshold, amount × MAD)
             → global soft threshold (sharpen_filter × MAD)
             × gain_i × spatial_weight_i
    gain_i = (amount_i / 200)^power × MAX_GAIN[i]
    spatial_weight_i = cosine S-curve inside the disk ellipse, feather = 2^i × eff
                       (Saturn: ∪ ring annulus mask)
    │
    ▼
Reconstruct: original + Σ contributions → PNG
```

### GUI Parameters → Internal Behavior

| GUI Parameter | Default | Internal Behavior |
|---|---|---|
| **L1 (0–500)** | 200 | ~2 px scale. `gain = (200/200)^1.0 × 29.15 = 29.15`. Finest detail |
| **L2 (0–500)** | 200 | ~4 px scale. `gain = 9.48`. Fine structures (belts, bands) |
| **L3 (0–500)** | 200 | ~8 px scale. `MAX_GAIN[2] = 0.0` → currently has no effect |
| **L4–L6 (0–500)** | 0 | ~16/32/64 px scale. `MAX_GAIN = 0.0` → no effect |
| **Denoise L1–L6** (Step 05) | 0.0 | Per-layer soft threshold before gain: `thr = amount × MAD(detail_i)`, measured on the brighter half of the image. 0.1 = WaveSharp's gentle default, 1.0 = strong. Useful on L1–L2 to suppress sensor noise |
| **Stretch** (Step 07) | Off | Contrast stretch of the preview PNG |
| **Saturation boost** (Step 07, color only) | On | Lab chroma boost of the preview PNG |
| **Apply color correction** (Step 07, color only) | On | Auto white balance + chromatic-aberration correction before sharpening |

> **What amount means**: `gain_i = (amount/200)^power × MAX_GAIN[i]`. amount = 200 → gain = MAX_GAIN, 400 → 2 × MAX_GAIN, 100 → 0.5 × MAX_GAIN. Because `MAX_GAIN[2:] = 0`, only L1 and L2 currently change the output.

### Internal Fixed Values

| Parameter | Value | Role |
|---|---|---|
| `MAX_GAINS` | [29.15, 9.48, 0, 0, 0, 0] | Per-level maximum gain, from OLS regression against WaveSharp reference output |
| `power` | 1.0 | `gain = (amount/200)^power × MAX_GAIN`. 1.0 = linear |
| `edge_feather_factor` | auto | Measured per image by `auto_wavelet_params()` |
| `disk_expand_px` | auto | `sqrt(rx×ry) × 0.0505`. The Otsu contour lands inside the visual limb because of limb darkening; this moves the feather start out to the real limb |
| `master_filter_type` | `"gaussian"` | B3-spline decomposition. Config also offers `"zerogauss"` and `"bilateral"` |

### À Trous B3-Spline Wavelet Decomposition

"À trous" (with holes) is an undecimated wavelet: instead of downsampling, it inserts zeros between kernel taps.

```
_B3 = [1, 4, 6, 4, 1] / 16   (5-tap separable kernel)

At level i, tap spacing = 2^i:
  smoothed_i  = B3_i ⊗ image_i    (reflect padding)
  detail_i    = image_i − smoothed_i
  image_{i+1} = smoothed_i
```

### Disk-aware Edge Feathering (Step 05)

Gain applies only inside the disk ellipse and fades toward the limb with a cosine S-curve; coarser levels use a wider feather:

```
feather_L = 2^L × edge_feather_factor
t = clip(dist_from_boundary / feather_L, 0, 1)
weight_L = 0.5 × (1 − cos(π × t))
```

The feather placement trades off a gray halo (gain reaches 0 too far inside the true limb) against a white-rim overshoot (full gain on the steep limb gradient). A white rim is treated as a **critical defect**, not an acceptable trade-off.

### Saturn Ring Sharpening Mask

For Saturn (`has_rings`), `compute_ring_sharpening_mask()` adds a true ring **annulus** to the gain mask (`master_ring_extension_enabled`, default **On**). It is built analytically from the ring radii, `pole_pa` and sub-observer latitude B. Only the front arc is used where the ring overlaps the globe, and the globe-to-ring gap gets **zero** gain. An earlier filled-ellipse version gave full gain to that gap (the globe's own PSF tail) and produced a white rim / dark trough at the globe-ring junction.

### Opt-in Experimental Options (config.py only)

All off by default, with no GUI control. See the status index in `WaveletConfig`'s docstring:

| Option | Status |
|---|---|
| `master_edge_extension_enabled` | Fills outside the disk ellipse before decomposition; only synthetic-test coverage |
| `master_coverage_aware_sharpening` | Scales gain by de-rotation coverage; measured negligible effect |
| `master_limb_darkening_confidence_enabled` | Scales gain by a Minnaert fit; unresolved |
| `master_navigation_limb_fit_enabled` | Saturn-only constrained limb fit; no verdict yet |
| `master_overshoot_clamp_radius_px` | **Do not enable**: removes 92–98% of real detail |
| `master_limb_fit_refinement_enabled` | **Do not enable**: introduces a white rim on Jupiter |
| `master_ring_scatter_subtraction_enabled` | **Do not enable**: introduces a dark-line / bright-arc artifact |

---

## 7. Step 06 — RGB Compositing

**Source**: `pipeline/modules/composite.py`, `pipeline/steps/rgb_composite.py`

```
Pass 1 — per window, per composite spec:
    (optional) Global Filter Normalize: per-(filter, window) disk-median scaling
        │
        ▼
    Channel stretch ("joint" when Stretch is on, otherwise "none")
        │
        ▼
    Reference channel: spec.align_ref > L > IR > R > G > B > CH4
    Shifts measured on the unsharpened step04 stacks (fallback: step05 images)
    by whole-frame gradient-magnitude NCC (sky floored at 8% of the bright
    end, σ1.5 Sobel magnitude, matchTemplate over ±Max Channel Shift,
    0.25 px grid refinement); rejected (no shift) if the peak is on the
    search boundary or NCC < 0.5
        [RGB]  non-reference channels → reference
        [LRGB] colour channels → R first, then the registered colour group
               → L as one unit (each channel shifted once, summed shift)
        │
        ▼
    np.stack([R, G, B])
        ├─ [RGB]  as-is
        └─ [LRGB] RGB → Lab, replace Lab L with the L channel, Lab → RGB
        │
        ▼
Pass 2 — per window:
    Global Normalize (cross-window mean luminance) → Saturation boost
    → Brightness Scale → 16-bit PNG
```

### Step 06 GUI Parameters → Internal Behavior (Mono mode)

| GUI Parameter | Default | Internal Behavior |
|---|---|---|
| **Max Channel Shift (px)** | 15.0 | A phase-correlation shift larger than this (in either axis) is never applied. Raise to 20–30 on nights with strong atmospheric dispersion |
| **Global Normalize** | On | After compositing, scales each window's composite so its mean luminance matches the cross-window average. Removes brightness flicker in the GIF |
| **Global Filter Normalize** | Off | Before compositing, computes the disk median for every (filter, window) and scales each window so all windows share the same median per filter. Pure scaling, so the dark background is preserved and nothing clips. Corrects transparency drift across the session |
| **Brightness Scale** | 1.0 | `output = composite × brightness_scale`, applied last. Range 0.1–2.0 |
| **Stretch** | Off | On: percentile stretch of the colour channels with one shared lo/hi (`"joint"`), so colour ratios are preserved |
| **Saturation boost** | On | Lab chroma gain so the 99.5th-percentile disk chroma reaches `headroom × 127`. Gain clamped to [1, 4]; never desaturates |
| **Composite Specs (R/G/B/L)** | RGB, IR-RGB, CH4-G-IR | Filter-to-channel mapping per composite. Setting L switches that composite to LRGB |

### Channel Alignment Quality Gate

`cv2.phaseCorrelate` can report a confident-looking but **wrong** peak when a channel has little signal correlated with the reference. On one real Jupiter session, the B channel against IR did this in ~25% of windows (e.g. dy = +2.16 where neighbouring windows measured −0.4 to −0.9), producing a limb colour fringe that the magnitude-only Max Shift gate does not catch. Each candidate shift is therefore kept only if the high-pass NCC (σ = 4 px) inside the disk (r < 0.95R) is higher **with** the shift than without it. If the disk cannot be detected, the old behaviour (trust the shift when within Max Shift) applies.

Only whole-frame correlation is used; disk-ROI cropping was tried and explicitly rejected.

### Internal Fixed Values

| Parameter | Value | Role |
|---|---|---|
| Reference channel priority | `align_ref` > L > IR > R > G > B > CH4 | Fixed, so the composite planet position does not jump between windows as channel brightness changes |
| `color_stretch_mode` (Stretch on) | `"joint"` | `"independent"` (per channel) is available in config |
| `saturation_headroom` | 0.15 | Target chroma = 0.15 × 127 at the 99.5th percentile |

### LRGB Compositing

When an L channel is specified, Lab luminance is replaced by the external channel:

```
Lab = cv2.cvtColor(rgb, COLOR_RGB2Lab)
Lab[:,:,0] = lrgb_weight × (L_external × 100) + (1−w) × Lab[:,:,0]
result = cv2.cvtColor(Lab, COLOR_Lab2RGB)
```

In IR-RGB, the sharper IR channel provides luminance detail while R/G/B provide colour.

### Post-composite Limb Desaturation — Disabled

A Lab a/b taper between 0.89r and 1.04r used to suppress limb colour fringes. It was **disabled on 2026-08-09**. Its ~10 px-wide band (outer 15% of the radius) was far wider than the ~1.5 px wavelength-dependent limb effect it claimed to fix, which suggests it had been tuned to hide channel misalignment (up to 14.6 px on Saturn). On real Saturn data it wiped out real colour across the outer disk, leaving a small coloured sphere inside a grey one. The code is kept, commented out, in `compose()`. If a thin fringe reappears, first check for a registration bug; only then re-enable with a much narrower band (e.g. 0.97r–1.02r).

### Color Camera Mode

There is no channel mapping. Each window gets automatic white balance (per-channel gain, G reference) and chromatic-aberration correction (R/B shift, G reference). The panel previews the first window's result before and after correction.

---

## 8. Step 08 — Animated GIF

**Source**: `pipeline/steps/gif.py`

```
step06_rgb_composite/ PNGs per composite (sorted by window time)
    │
    ▼
Planet centering: detect each frame's disk centre → moving median over
5 frames → shift every frame so the smoothed centre sits at the image centre
    │
    ▼
uint8 conversion → optional resize (cv2 INTER_LANCZOS4)
    │
    ▼
Per-frame Floyd–Steinberg dithering to a 256-colour palette
    │
    ▼
{composite}_animation.gif (loop = 0, infinite repeat)
```

Centering removes the frame-to-frame jitter of the composited planet. The moving median keeps blurry-limb detection noise from making features such as the GRS appear to wobble.

### GUI Parameters → Internal Behavior

| GUI Parameter | Default | Internal Behavior |
|---|---|---|
| **FPS** | 6.0 | `frame_duration = int(1000 / fps)` [ms], passed as Pillow's `duration` |
| **Resize Factor** | 1.0 | `new_size = (int(w × factor), int(h × factor))`, LANCZOS4 resampling |

---

## 9. Step 09 — Summary Grid

**Source**: `pipeline/steps/summary_grid.py`

Step 09 always produces `summary_grid_simple.png` and, when in mono mode with Step 05 output present, additionally produces `summary_grid.png` (two-zone) and optionally per-window analytic PNGs in `analytic/`.

### Output Files

| File | Contents | Condition |
|------|----------|-----------|
| `summary_grid_simple.png` | Composites only (all windows × all composites) | Always |
| `summary_grid.png` | Composites (left zone) + Step 05 filter images (right zone), same cell size, vertical divider | Mono mode + Step 05 data exists |
| `analytic/window_XX_analytic.png` | Per-window detailed view (see below) | `save_analytic=True`, mono mode |

### Simple Grid (`summary_grid_simple.png`)

```
Step 06 composite PNGs (all windows)
    │
    ▼
Per image:
    Black point: pixel = clip((p − bp) / (1 − bp), 0, 1)
    Gamma:       pixel = pixel ^ (1 / gamma)
    Resample to cell_size
    │
    ▼
Grid layout (rows = windows, columns = composites) → PNG
```

### Two-Zone Grid (`summary_grid.png`)

```
Left zone: Step 06 composites (cell_size × cell_size each)
Right zone: Step 05 filter PNGs (same cell_size)
    │
    ▼
Vertical divider between zones
Column labels above each image
Row (window) time labels on the left
    │
    ▼
summary_grid.png
```

Both zones use `cell_px` = the configured **Cell Size** value. The filter zone width = `n_filters × cell_px + gaps`.

### Analytic View (`analytic/window_XX_analytic.png`)

One PNG per time window. Layout (top to bottom):

```
[Header: window time range]

[Filter images row]           ← Step 05 PNGs, cell_size each
[Filter stats block]          ← Frames / Q.Post / Stab. / Stacked per filter, column-aligned to images above

─────────────────────────────────── (divider)

[Composite images row]        ← Step 06 PNGs, cell_size each
[Rotation indicators]         ← N/S pole axis segments + rotation-direction arrow drawn
                                 outside the disk on the first composite image
[Align table]                 ← rows = filter names; columns = composites
                                 cell = "[role] shift" or "[role] ref" or "—"
[Sat row]                     ← saturation boost per composite

─────────────────────────────────── (separator)

[Global params line]          ← Win.Q / Rot / Wvl / bp / γ
```

**Rotation indicators**: Short line segments outside the disk limb mark the north (N, blue) and south (S, red) pole directions based on the logged `pole_pa_deg` and `tracker_flip_ns` values. A curved arrow drawn outside the disk indicates the prograde rotation direction.

#### Filter Stats Block

Drawn above the divider, x-aligned to filter image columns. No duplicate filter name header (filter names already appear as image labels above).

| Row | Value |
|-----|-------|
| **Frames** | `n_used / n_total` (frames passing quality threshold / all frames in window) |
| **Q.Post** | Mean quality score of retained frames after σ-clipping, 0–1 |
| **Stab.** | `1 / (1 + CV)` where `CV = std/mean` of per-frame quality scores |
| **Stacked** | Final frame count actually summed in the lucky stack |

#### Align Table

- **Rows** = filter names (IR, R, G, B, CH4 …) derived from `CompositeSpec` fields across all composites
- **Columns** = composite names (RGB, IR-RGB, …)
- **Cell value** = `[role] shift` where `role` ∈ {L, R, G, B} and `shift` = `(Δx, Δy)` from `composite_log.json`; `ref` if the filter was the reference channel; `—` if the filter is not used in that composite
- **Alignment keys** in `composite_log.json` are filter names (IR/R/G/B/CH4), not channel roles

#### Canvas Height Pre-calculation

Height is computed before `Image.new()` using a 1×1 probe draw:

```python
canvas_h = (pad + header_h
            + filter_lbl_h + filter_px   # filter images
            + fstats_h                   # filter stats (above divider)
            + section_gap                # divider
            + comp_lbl_h + comp_px       # composite images (name only label)
            + apar_h                     # align table + separator + global params
            + pad)
```

`label_margin` (width of widest row label + 12 px) is added to `canvas_w` to prevent row labels from overflowing the left edge.

### GUI Parameters → Internal Behavior

| GUI Parameter | Default | Internal Behavior |
|---|---|---|
| **N Best Windows** | 0 | Number of windows to include in the grid, selected by descending `quality_score`. 0 = include all enumerated windows. When > 0, a greedy non-overlapping selection algorithm picks the top-N windows: candidates are sorted by quality score and accepted in order, skipping any candidate whose time range overlaps an already-accepted window |
| **Allow Window Overlap** | Off | When Off (default), the greedy selection described above enforces non-overlap — each accepted window's time range must not intersect any previously accepted window. When On, all top-N windows by score are accepted regardless of temporal overlap |
| **Black Point** | 0.04 | `pixel = clip((p − 0.04) / (1 − 0.04), 0, 1)`. Pushes background noise to pure black. 0.02–0.08 recommended |
| **Gamma** | 0.9 | `pixel = pixel ^ (1/0.9) ≈ pixel ^ 1.11`. <1.0=brighter (0.9 default slightly brightens the planet), >1.0=darker, 1.0=no change |
| **Cell Size (px)** | 300 | Each composite and filter image in the grid is resampled to this size. Both zones use the same cell size in the two-zone grid |
| **Save Analytic View** | True | When True, generates `analytic/window_XX_analytic.png` for each time window. Mono mode only |

---

## 10. Common Module: Disk Detection (find_disk_center)

**Source**: `pipeline/modules/derotation.py`

Used by Steps 02, 04, 05 and 06.

```
Phase 1 — Centre and shape (Otsu binary method)
1. arr8 = clip(image × 255, 0, 255).uint8
2. effective_thresh = Otsu × (1 − 0.10)
   (margin_factor = 0.10: lowers the threshold to include the dark limb)
3. Morphological closing (7×7 elliptical kernel) → fill small gaps
4. Largest contour → cv2.fitEllipse → (cx, cy, semi_a, semi_b, angle)

Phase 1b — Ringed blob (raw aspect < 0.80, e.g. Saturn)
5. Re-threshold the blob at its 60th brightness percentile to isolate the
   bright disk core and refit the ellipse to the core.
6. If no plausible core is found (e.g. Saturn CH4, where the disk is darker
   than the rings): radial limb search for semi_a instead; shape is marked
   unreliable (shape_reliable = False) and borrowed from a sibling filter
   in Step 04.

Phase 2 — Radius refinement (gradient limb detection)
7. Cast 72 rays from (cx, cy), every 5°; sample each over
   [0.75, 1.30] × semi_a  (ring-stripped core: [0.6, 1.8])
8. 1-D Gaussian smoothing (σ = 1.5 px) → steepest descent → parabolic
   sub-pixel refinement
9. Reject outliers (median ± 2σ) → median → semi_a_refined
   (semi_b rescaled to keep the fitted aspect ratio)

Return: (cx, cy, semi_a_refined, semi_b, angle_deg)
```

The internal `_find_disk_center_impl()` also returns a **confidence** value: 1.0 for a ringless fit, (0, 1] for successful core isolation, 0.5 / 0.3 for the radial-limb fallback with / without ≥ 8 confirmed edges, 0.0 for a failed fit. It also returns a `shape_reliable` flag. Step 04 uses these to decide when a filter must borrow shape or pose from a sibling filter.

The binary method locates the centre accurately but underestimates the radius, because Otsu clips the dark outer limb. The gradient method finds the actual intensity inflection on each ray. Even that edge sits inside the true geometric limb because of limb darkening, which is why satellite plate scale uses a separate Minnaert-corrected radius (§5).

---

## 11. Common Module: Sub-pixel Alignment

**Source**: `pipeline/modules/derotation.py`

### apply_shift

```python
M = np.float32([[1, 0, dx], [0, 1, dy]])
cv2.warpAffine(image, M, (w, h), flags=INTER_CUBIC, borderMode=BORDER_REPLICATE)
```

- **INTER_CUBIC**: bicubic interpolation (not LANCZOS4, which rings at the limb)
- **BORDER_REPLICATE**: replicates edge pixels (BORDER_CONSTANT = 0 would bleed black into the image edge)

### apply_shift_and_scale

```
output = ref_center + scale × (input − target_center)
```

Scales about the frame's **own** measured centre, then moves that centre onto the reference centre. This is not the same as scaling about the reference centre and then translating; the two agree only at scale = 1. Used by Step 04's pre-warp alignment: a 1–2% frame-to-frame radius variation is under 1 px at the disk but 1.6–2.8 px at Saturn's Cassini Division, enough to wash it out in a stack. A frame-to-frame rotation correction was tried and reverted, because the measured 3–4° angle swing proved to be contour-fit noise.

### subpixel_align (phase correlation)

```python
(raw_dx, raw_dy), _ = cv2.phaseCorrelate(ref_f32, tgt_f32)
return -raw_dx, -raw_dy
```

`cv2.phaseCorrelate` returns the forward displacement of the target relative to the reference, not the correction. The result is **negated** so that `apply_shift(target, *subpixel_align(ref, target))` aligns the target. Precision is ~0.1 px.

### limb_center_align

Detects the disk centre in the target with `find_disk_center()` and returns `(ref_cx − cx, ref_cy − cy)`. Used for per-frame global alignment in Step 02, and as a post-warp fallback in Step 04.

---

*References: Starck & Murtagh (2006), Bijaoui (1991), Mackay (2013 arXiv:1303.5108), Zack (1977)*
