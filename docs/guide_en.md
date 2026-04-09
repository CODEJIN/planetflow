# Planetary Imaging Post-Processing Pipeline — User Guide

> **Version**: Current Development Version  
> **Target Users**: Advanced planetary imagers familiar with Firecapture, AutoStakkert!4, and WinJUPOS

---

## Table of Contents

1. [Overview](#1-overview)
2. [Main Window Layout](#2-main-window-layout)
3. [Global Settings](#3-global-settings)
4. [Step 01 — PIPP Preprocessing](#4-step-01--pipp-preprocessing)
5. [Step 02 — AutoStakkert!4](#5-step-02--autostakkert4)
6. [Step 03 — Wavelet Preview Sharpening](#6-step-03--wavelet-preview-sharpening)
7. [Step 04 — Quality Assessment & Window Detection](#7-step-04--quality-assessment--window-detection)
8. [Step 05 — De-rotation Stacking](#8-step-05--de-rotation-stacking)
9. [Step 06 — Wavelet Master Sharpening](#9-step-06--wavelet-master-sharpening)
10. [Step 07 — RGB Composite (Master)](#10-step-07--rgb-composite-master)
11. [Step 08 — Time-Series RGB Composite](#11-step-08--time-series-rgb-composite)
12. [Step 09 — Animated GIF](#12-step-09--animated-gif)
13. [Step 10 — Summary Grid](#13-step-10--summary-grid)
14. [Run All](#14-run-all)
15. [Output Folder Structure](#15-output-folder-structure)
16. [Quick Parameter Reference](#16-quick-parameter-reference)

---

## 1. Overview

This tool automates the planetary imaging post-processing pipeline. Starting from SER videos preprocessed with PIPP, it accepts AutoStakkert!4 stacking results and guides you through wavelet sharpening → quality assessment → de-rotation stacking → RGB compositing → time-series animation → summary grid generation.

### Complete Workflow

```
Raw SER Videos (from Firecapture)
         │
         ▼
[Step 01] PIPP Preprocessing     ← SER → Cropped SER (Optional)
         │
         ▼
[Step 02] AutoStakkert!4         ← Manual external execution
         │
         ▼
[Step 03] Wavelet Preview        ← TIF → Sharpened PNG (Required)
         │
         ▼
[Step 04] Quality Assessment     ← Optimal time window detection (Required)
         │
         ▼
[Step 05] De-rotation Stacking   ← Rotation correction + stacking (Required)
         │
         ▼
[Step 06] Wavelet Master         ← Master image sharpening (Required)
         │
         ▼
[Step 07] RGB Composite (Master) ← Filter channel compositing (Required)
         │
         ├───────────────────────────────────────────────────┐
         ▼                                                   ▼
[Step 08] Time-Series Composite  ← Per-epoch compositing (Optional)
         │
         ▼
[Step 09] Animated GIF           ← Rotation time-series animation (Optional)

[Step 10] Summary Grid           ← Step 07 results → Levels + grid (Optional)
```

---

## 2. Main Window Layout

<!-- TODO: Insert full main window screenshot -->
*Figure 2-1: Main window overall layout*

### 2.1 Left Sidebar

The left side of the screen contains the step navigation list.

| Element | Description |
|---------|-------------|
| **⚙ Settings** | Opens the global settings panel. Configure planet preset, camera mode, and filter list. |
| **Step List** | Click Step 01–Step 10 to navigate to the corresponding panel. |
| **Optional** | Steps marked as optional can be skipped. |
| **Separators** | There are separators before Step 03 and Step 08. Steps 01–02 are external tool integration; Steps 03–07 are core processing; Steps 08–10 are additional outputs. |

### 2.2 Right Main Area

| Element | Description |
|---------|-------------|
| **Panel Area** | Displays the settings for the selected step. |
| **Log Area** | Pipeline execution logs are displayed at the bottom. |

### 2.3 Status Bar

The bottom status bar shows the current output folder path and pipeline readiness status.

### 2.4 Common Buttons

Each step panel has the following buttons at the bottom:

| Button | Description |
|--------|-------------|
| **Run** | Executes only the current step. |
| **Next Step →** | After running, automatically navigates to the next step panel. Not available on Step 10 (the last step). |

---

## 3. Global Settings

<!-- TODO: Insert Settings panel screenshot -->
*Figure 3-1: Global Settings panel*

Global settings define the base values that affect the entire pipeline. Always review these before starting a session.

### 3.1 Planet Presets

| Preset | Target Name | Horizons ID | Rotation Period |
|--------|-------------|-------------|-----------------|
| **Jupiter** | Jup | 599 | 9.9281 h |
| **Saturn** | Sat | 699 | 10.56 h |
| **Mars** | Mar | 499 | 24.6229 h |
| **Custom** | User input | User input | User input |

Selecting a preset automatically fills in the fields below.

### 3.2 Parameter Details

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Planet Preset** | Jupiter | Select the target planet. When Custom is selected, the three fields below must be entered manually. |
| **Target Name** | Jup | Short identifier used in filenames and logs within the pipeline. |
| **Horizons ID** | 599 | NASA JPL Horizons service body ID. Used by Step 05 de-rotation to automatically fetch the planet's north pole angle (NP.ang) for the observation date. |
| **Rotation Period (h)** | 9.9281 | The planet's rotation period in hours. This is the basis for de-rotation calculations in Step 05. |
| **Camera Mode** | Mono | **Mono**: Uses a filter wheel, separate SER files per filter. **Color**: Single color camera with Bayer RGB output. Selecting Color automatically sets the filter list to `COLOR` and disables editing. |
| **Filter List** | IR,R,G,B,CH4 | Comma-separated list of filters used. This list populates the channel dropdowns in Step 07 compositing. Automatically set to `COLOR` when Color camera mode is selected. |
| **Language** | ko | Interface language. Changes take effect after restarting the application. |

> **Tip**: Settings are saved per session. When you reopen the tool, your previous configuration is automatically restored.

---

## 4. Step 01 — PIPP Preprocessing

<!-- TODO: Insert Step 01 panel screenshot (with preview) -->
*Figure 4-1: Step 01 panel — Left: form, Right: SER frame preview*

Uses PIPP (Planetary Imaging PreProcessor) to crop SER videos centered on the planet and extract the Region of Interest (ROI).

> **Optional Step**: This step can be skipped if your SER files are already cropped or if you've run PIPP separately.

### 4.1 Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **SER Video Folder** | (Required) | — | Path to the folder containing SER files. Automatically searches all subfolders for `.SER` files. Browse with the `...` button or type the path directly. |
| **Output Folder** | Auto-set | — | Folder where PIPP-processed SER files will be saved. Automatically set under the global output base folder. |
| **ROI Size (px)** | 448 | 64–1024 (step 16) | Square crop size for PIPP output. Set this large enough to encompass the planetary disc. 448–512px is typical for Jupiter. If too small, the planet may be clipped. |
| **Min Disc Diameter (px)** | 50 | 10–500 (step 5) | Minimum disc size to be considered a valid planet detection. Frames where the detected disc is smaller than this value are treated as bad frames and discarded. Used to filter frames where atmospheric turbulence makes the planet appear very blurry or small. |

### 4.2 Live Preview

The right panel shows a live preview.

- **Left panel (Original + Detection)**: Shows a representative frame extracted from the SER file with planet detection overlays.
  - **Cyan box (Planet)**: Automatically detected planet region
  - **Green box (ROI)**: The area that will be cropped to the set ROI size
- **Right panel (ROI Crop)**: Shows the actual cropped output to be produced.

The preview automatically refreshes when ROI size or minimum diameter changes.

> **Note**: The ROI box (green) must be substantially larger than the planet box (cyan). A warning is displayed if the ROI box extends beyond the image boundary.

---

## 5. Step 02 — AutoStakkert!4

<!-- TODO: Insert Step 02 panel screenshot -->
*Figure 5-1: Step 02 panel — AutoStakkert!4 manual execution guide*

AutoStakkert!4 is an external program that cannot be executed directly by the pipeline. This step is an informational panel guiding you on how to run AS!4.

> **Optional Step**: Skip this step if AS!4 stacking is already complete.

### How to Proceed

1. Follow the instructions in this panel and **run AutoStakkert!4 separately**.
2. Use Step 01's output folder (PIPP-processed SER files) as the AS!4 input.
3. Once AS!4 stacking is complete, click **"Done"** to proceed to the next step.

The path to AS!4's output TIF files is specified directly in Step 03.

---

## 6. Step 03 — Wavelet Preview Sharpening

<!-- TODO: Insert Step 03 panel screenshot (wavelet sliders emphasized) -->
*Figure 6-1: Step 03 panel — Wavelet sharpening configuration*

Applies wavelet sharpening to TIF files output by AutoStakkert!4 and converts them to PNG format. These PNGs are used as input for Step 04 quality assessment.

> **Required Step**: This step cannot be skipped.

### 6.1 Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **AS!4 TIF Folder** | (Required) | — | The folder where AutoStakkert!4 saved TIF files. Processes all `.tif` / `.TIF` files in this folder. Browse with the `...` button or type the path. |
| **Output Folder** | Auto-set | — | Folder where wavelet-processed PNG files will be saved. Automatically set under the same path as the TIF input folder when selected. |
| **Border Taper (px)** | 0 | 0–100 (step 5) | Applies a soft cosine fade to image edges. 0 = disabled (recommended). Only use this when severe ringing artifacts appear at the edges after wavelet processing. |

### 6.2 Wavelet Levels (L1–L6)

| Level | Default | Range | Characteristics |
|-------|---------|-------|-----------------|
| **L1** | 200 | 0–500 | Finest detail (pixel-level sharpening) |
| **L2** | 200 | 0–500 | Fine detail |
| **L3** | 200 | 0–500 | Medium detail |
| **L4** | 0 | 0–500 | Large-scale structures (risk of noise amplification) |
| **L5** | 0 | 0–500 | Even larger structures |
| **L6** | 0 | 0–500 | Largest structures |

- Sliders and number inputs are **bidirectionally synchronized**.
- It is recommended to **only activate L1–L3** for planetary imaging. L4 and above can excessively amplify noise.
- These values apply to the preview sharpening in Step 03. Final master image sharpening is configured separately in Step 06.

> **WaveSharp Compatibility**: These sliders use the same 0–200 scale as WaveSharp software's Amount values (default 200).

---

## 7. Step 04 — Quality Assessment & Window Detection

<!-- TODO: Insert Step 04 panel screenshot -->
*Figure 7-1: Step 04 panel — Quality assessment configuration*

Automatically evaluates the image quality of each TIF frame and detects the optimal time window for stacking.

> **Required Step**: This step cannot be skipped.

### 7.1 Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **Input Folder** | Auto-set | — | Automatically set to the same AS!4 TIF folder as Step 03. |
| **Output Folder** | Auto-set | — | Quality score CSV and window recommendation JSON are saved here. |
| **Window Length (seconds)** | 900 | 30–3600 (step 30) | Length of the optimal time window. Specifies the time range of frames to process while the planet rotates. **Jupiter**: 900s (15 min). **Mars/Saturn**: 1,200–1,800s (20–30 min). |
| **Filter Cycle (seconds)** | 270 | 10–600 (step 15) | Time in seconds for one complete filter cycle (IR→R→G→B→CH4→IR). Must be set accurately to match your actual capture pattern, as it's used for channel grouping in Step 08 time-series compositing. |
| **Number of Windows** | 1 | 1–10 | Number of optimal windows to detect. **1**: Find only the single best window (for Step 05 stacking). **2–3**: Detect multiple windows at different epochs (for Step 08 time-series). |
| **Allow Overlap** | Off | — | Checked: Detected windows may overlap in time. Unchecked: Each window is non-overlapping (default). Check this when trying to extract multiple windows from a short dataset. |
| **Min Quality Threshold** | 0.0 | 0.0–1.0 (step 0.05) | Frames below this quality score are excluded from window optimization. 0.0 = include all frames. 0.2–0.3 = remove bad frames. **Setting too high may leave too few valid frames.** |

### 7.2 Output Files

- `quality_scores.csv`: Quality score for each TIF file
- `windows.json`: Detected optimal time window information
- `windows_summary.txt`: Human-readable window summary

---

## 8. Step 05 — De-rotation Stacking

<!-- TODO: Insert Step 05 panel screenshot -->
*Figure 8-1: Step 05 panel — De-rotation stacking configuration*

Stacks frames within the optimal windows detected in Step 04, correcting for planetary rotation during the process.

> **Required Step**: This step cannot be skipped.

### 8.1 Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **Input Folder** | Auto-set | — | Automatically set to the same AS!4 TIF folder as Step 04. |
| **Output Folder** | Auto-set | — | De-rotation stacked master TIF files are saved here. |
| **Rotation Period (h)** | 9.9281 | 1.0–50.0 (step 0.01) | The planet's rotation period in hours. Automatically filled from global settings. **Jupiter**: 9.9281h, **Saturn**: 10.56h, **Mars**: 24.6229h. An inaccurate value will cause blurring of fine details. |
| **Warp Scale** | 0.20 | 0.0–2.0 (step 0.01) | Spherical distortion correction strength. ~0.2 is typical. Important for stacking frames near the equator of oblate planets like Jupiter. 0.0 = no correction. |
| **Min Quality Threshold** | 0.3 | 0.0–1.0 (step 0.05) | Frames below this quality score are excluded from stacking. Raise to 0.3–0.5 when seeing conditions are poor to more strictly filter bad frames. |
| **Normalize Brightness** | Off | — | Normalizes the brightness of each frame before stacking. Enable this when frames have significant brightness variations due to changing seeing conditions. Generally recommended off for typical conditions. |

### 8.2 JPL Horizons Integration

Step 05 automatically queries the NASA JPL Horizons API to retrieve the planet's north pole angle (NP.ang) at the time of observation. The **Horizons ID** in global settings must be set correctly. An internet connection is required.

---

## 9. Step 06 — Wavelet Master Sharpening

<!-- TODO: Insert Step 06 panel screenshot -->
*Figure 9-1: Step 06 panel — Master image wavelet sharpening*

Applies wavelet sharpening to the master TIF images generated by Step 05. Since master images result from stacking thousands of frames, their SNR is extremely high, allowing stronger sharpening than Step 03.

> **Required Step**: This step cannot be skipped.

### 9.1 Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **Input Folder** | Auto-set | — | Automatically set to the Step 05 de-rotation result TIF folder. |
| **Output Folder** | Auto-set | — | Wavelet-processed master PNG files are saved here. |

### 9.2 Wavelet Levels (L1–L6)

Same structure as Step 03, but **stronger values can be safely used** since it's applied to master images.

| Level | Default | Recommended Range | Characteristics |
|-------|---------|-------------------|-----------------|
| **L1** | 200 | 100–400 | Highest-resolution pixel-level detail |
| **L2** | 200 | 100–400 | Fine structures (belts, streaks) |
| **L3** | 200 | 50–300 | Medium-scale structures |
| **L4** | 0 | 0–100 | Large-scale tonal contrast |
| **L5** | 0 | 0 | Not recommended |
| **L6** | 0 | 0 | Not recommended |

> **Tip**: Master images have very high SNR, so raising L1 and L2 to 300–400 rarely produces artifacts. Tune Step 03 and Step 06 values independently for optimal results.

---

## 10. Step 07 — RGB Composite (Master)

<!-- TODO: Insert Step 07 panel screenshot (composite table + radio buttons + live preview) -->
*Figure 10-1: Step 07 panel — RGB composite settings with live preview*

Combines per-filter master PNGs from Step 06 into color images. Multiple composite types (RGB, LRGB, false color) can be defined and produced in a single session.

> **Required Step**: This step cannot be skipped.

### 10.1 Base Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **Input Folder** | Auto-set | — | Automatically set to the Step 06 master PNG folder. |
| **Output Folder** | Auto-set | — | RGB composite result PNGs are saved here. |
| **Max Channel Shift (px)** | 15.0 | 0.0–100.0 | Maximum allowed shift distance in channel-to-channel phase correlation alignment. If the computed shift exceeds this value, alignment is not applied (prevents runaway misalignment). Raise to 20–30 on nights with strong atmospheric dispersion. |

### 10.2 Composite Specification Table

Each row defines one composite output image.

| Column | Description |
|--------|-------------|
| **👁 (Radio Button)** | Selects which composite to display in the preview. Only one can be selected at a time. |
| **Name** | The name of the composite image. Used in the output filename (e.g., `RGB_composite.png`). |
| **R Channel** | Filter assigned to the red channel. |
| **G Channel** | Filter assigned to the green channel. |
| **B Channel** | Filter assigned to the blue channel. |
| **L Channel** | Filter assigned as luminance (brightness) channel. Selecting this enables **LRGB compositing** mode. `──` = not used (plain RGB). |
| **✕** | Deletes this composite specification row. |

#### Default Composite Specs

| Name | R | G | B | L | Description |
|------|---|---|---|---|-------------|
| **RGB** | R | G | B | (none) | Standard 3-color composite |
| **IR-RGB** | R | G | B | IR | LRGB using IR as luminance. The IR channel's high resolution enhances fine luminance detail. |
| **CH4-G-IR** | CH4 | G | IR | (none) | Methane band false-color composite. Emphasizes Jupiter's cloud structure and the GRS. |

#### + Add Button

Adds a new composite specification. You can freely create any filter combination you need, such as `UV-RGB` (R=UV, G=R, B=B) or other custom mappings.

### 10.3 Live Preview

The right panel shows a live preview of the currently selected (radio button) composite.

- **Left panel (Input Channel)**: Grayscale image of the reference channel (R for RGB mode, L for LRGB mode)
- **Right panel (Composite Result)**: The result of combining the configured R/G/B(/L) channels

The preview automatically refreshes 400ms after changing R/G/B/L channel dropdowns.

> **Note**: The preview is computed without channel alignment (phase correlation) for speed. The effect of channel alignment can be verified after running the step by checking the Step 10 preview.

---

## 11. Step 08 — Time-Series RGB Composite

<!-- TODO: Insert Step 08 panel screenshot -->
*Figure 11-1: Step 08 panel — Time-series composite settings*

Uses the wavelet preview PNGs from Step 03 to create time-series RGB composite images. Used for planetary rotation time-series analysis.

> **Optional Step**: Skip if time-series compositing is not needed.

### 11.1 How It Works

1. Scans PNG files in the Step 03 output folder.
2. Groups frames into time-based buckets using the **filter cycle time** set in Step 04.
3. Applies the same compositing settings as Step 07 to each group.
4. Results are saved in `step08_series/` in chronological order.

Step 08 uses Step 03/04/07 settings automatically with no additional parameter input.

---

## 12. Step 09 — Animated GIF

<!-- TODO: Insert Step 09 panel screenshot -->
*Figure 12-1: Step 09 panel — GIF animation configuration*

Combines the time-series composite results from Step 08 into a planetary rotation animation GIF.

> **Optional Step**: Only available when Step 08 has been run.

### 12.1 Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **Input Folder** | Auto-set | — | Automatically set to the Step 08 time-series PNG folder. |
| **Output Folder** | Auto-set | — | GIF file is saved here. |
| **FPS** | 6.0 | 1.0–30.0 (step 0.5) | GIF playback speed in frames per second. **6–10 FPS** is typical for planetary rotation animations. Lower values produce slower playback. |
| **Resize Factor** | 1.0 | 0.1–2.0 (step 0.1) | Output GIF size multiplier. **1.0** = original size, **0.5** = half size (reduces file size), **2.0** = double size. Use 0.5 to reduce GIF file size for web posting. |

---

## 13. Step 10 — Summary Grid

<!-- TODO: Insert Step 10 panel screenshot (with levels preview) -->
*Figure 13-1: Step 10 panel — Summary grid with levels adjustment preview*

Applies levels correction to Step 07 RGB composite results and combines them into a single summary grid image. Used for generating final images for observation reports or forum posts.

> **Optional Step**: Use only when needed.

### 13.1 Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **Input Folder** | Auto-set | — | Automatically set to the Step 07 RGB composite result folder. |
| **Output Folder** | Auto-set | — | Summary grid PNG is saved here. |
| **Black Point** | 0.04 | 0.0–0.5 (step 0.01) | Pixels at or below this value are remapped to pure black (0). Suppresses background sky noise and gives the planet a clean dark border. Recommended range: **0.02–0.08**. |
| **White Point** | 1.0 | 0.5–1.0 (step 0.01) | Pixels at or above this value are clipped to pure white (1.0). Used to clip over-saturated pixels. Typically kept at 1.0. |
| **Gamma** | 0.9 | 0.1–3.0 (step 0.05) | Brightness gamma correction. **1.0** = no correction / **< 1.0** = brighter (typically 0.8–1.0 recommended) / **> 1.0** = darker. |
| **Cell Size (px)** | 300 | 100–600 (step 50) | Size in pixels of each composite image cell within the summary grid. |

### 13.2 Live Preview

The right panel shows before/after levels adjustment previews.

- **Before (레벨 조정 전)**: Original Step 07 output image
- **After (레벨 조정 후)**: Image with black point + gamma applied

The preview automatically refreshes 400ms after changing any parameter.

---

## 14. Run All

<!-- TODO: Insert Run All button / running state screenshot -->
*Figure 14-1: Pipeline running state*

Clicking the **"Run All"** button in the bottom of the left sidebar automatically runs Steps 03 through 10 in sequence.

- Steps 01 and 02 are excluded from automatic execution as they depend on external tools (PIPP, AS!4).
- Optional steps are included or excluded based on your settings.
- If an error occurs during execution, the pipeline halts at that step and an error message is printed to the log.

---

## 15. Output Folder Structure

After pipeline execution, the following folders are created under the output base folder (e.g., `260402_output/`):

```
{output_base}/
├── step03_wavelet_preview/     # Step 03: Wavelet-processed preview PNGs
│   ├── 2026-03-20-1046_1-U-IR-Jup_..._wavelet.png
│   └── ...
├── step04_quality/             # Step 04: Quality assessment results
│   ├── quality_scores.csv
│   ├── windows.json
│   └── windows_summary.txt
├── step05_derotate_stack/      # Step 05: De-rotation master TIFs
│   └── window_01/
│       ├── IR_master.tif
│       ├── R_master.tif
│       └── ...
├── step06_wavelet_master/      # Step 06: Wavelet master PNGs
│   └── window_01/
│       ├── IR_master.png
│       ├── R_master.png
│       └── ...
├── step07_rgb_composite/       # Step 07: RGB composite PNGs
│   └── window_01/
│       ├── RGB_composite.png
│       ├── IR-RGB_composite.png
│       ├── CH4-G-IR_composite.png
│       └── composite_log.json
├── step08_series/              # Step 08: Time-series composite PNGs
│   ├── 2026-03-20T10:46_RGB.png
│   └── ...
├── step09_gif/                 # Step 09: Animated GIF
│   └── RGB_animation.gif
└── step10_summary_grid/        # Step 10: Summary grid PNG
    └── summary_grid.png
```

---

## 16. Quick Parameter Reference

### Recommended Settings by Planet

| Parameter | Jupiter | Saturn | Mars |
|-----------|---------|--------|------|
| Rotation Period (h) | 9.9281 | 10.56 | 24.6229 |
| Horizons ID | 599 | 699 | 499 |
| Window Length (s) | 900 | 1,200–1,800 | 1,200–1,800 |
| ROI Size (px) | 448–512 | 512–640 | 256–384 |

### Recommended Settings by Seeing Conditions

| Condition | Min Quality (Step 04/05) | Wavelet L1/L2 (Step 06) |
|-----------|--------------------------|--------------------------|
| Good | 0.0–0.1 | 200–400 |
| Average | 0.2–0.3 | 150–250 |
| Poor | 0.4–0.5 | 100–200 |

### Filter Channel Quick Reference (Mono Camera)

| Filter | Wavelength | Typical Use |
|--------|------------|-------------|
| **IR** | ~685–1000nm | Luminance channel (LRGB), best atmospheric penetration |
| **R** | ~600–700nm | Red channel, RGB/LRGB compositing |
| **G** | ~500–600nm | Green channel |
| **B** | ~400–500nm | Blue channel, most affected by atmospheric dispersion |
| **CH4** | ~889nm | Methane absorption band, false-color compositing |

---

*Screenshots for this guide need to be added by the user. Save images in the `docs/images/` folder and replace each `<!-- TODO -->` comment with the actual image link.*
