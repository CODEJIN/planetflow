"""
RGB / LRGB compositing module.

Supported composite types (controlled by CompositeSpec in config):
  - RGB:      direct R/G/B channel combination
  - LRGB:     IR (or any filter) as luminance, RGB as colour (Lab space blend)
  - False colour: any filter-to-channel mapping (e.g. CH4→R, G→G, IR→B)

Channel alignment:
  All channels are aligned to the reference channel (the L channel if present,
  else IR → R → G → B) by whole-frame gradient-magnitude NCC, measured on the
  unsharpened step04 stacks when available (see estimate_channel_shift). With
  an L channel, the colour channels are registered to each other first and
  then moved to L as one group (see align_lrgb). This corrects the tracking
  drift between filter captures, atmospheric dispersion and filter offsets.

Auto-stretch:
  Each input channel is independently auto-stretched using percentile
  normalisation before compositing.  This matches common practice in
  planetary imaging workflows (each filter has different sky background levels).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from pipeline.modules.derotation import apply_shift, find_disk_center


# ── Per-channel stretch ────────────────────────────────────────────────────────

def auto_saturate(
    img: np.ndarray,
    phigh: float = 99.5,
    headroom: float = 0.30,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Auto-boost saturation in Lab space.

    Computes gain so that p(phigh) of disk chroma reaches headroom × 127
    (Lab max chroma).  gain is clamped to [1.0, 4.0] so the function never
    desaturates and never amplifies more than 4×.

    Args:
        img:      float32 [0,1] RGB (H,W,3).
        phigh:    Chroma percentile used as reference (mirrors stretch_phigh).
        headroom: Fraction of Lab max chroma (127) that p(phigh) should reach.
                  Data-adaptive: low-saturation images get more boost, high-
                  saturation images get less, without needing per-dataset tuning.
        mask:     Boolean mask selecting pixels to sample (e.g. disk interior).
    Returns:
        Saturation-boosted float32 [0,1] RGB image.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    a, b = lab[:, :, 1], lab[:, :, 2]
    chroma = np.sqrt(a ** 2 + b ** 2)

    sample = chroma[mask] if (mask is not None and mask.any()) else chroma.ravel()
    current = float(np.percentile(sample, phigh))
    if current < 1e-3:
        return img

    target = 127.0 * headroom
    gain = float(np.clip(target / current, 1.0, 2.0))
    lab[:, :, 1] = np.clip(a * gain, -127.0, 127.0)
    lab[:, :, 2] = np.clip(b * gain, -127.0, 127.0)
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0.0, 1.0).astype(np.float32)


def auto_stretch(
    img: np.ndarray,
    plow: float = 0.1,
    phigh: float = 99.9,
    target_hi: float = 1.0,
) -> np.ndarray:
    """Percentile-based linear stretch.  p(phigh) maps to target_hi; values above can reach 1.0."""
    lo, hi = np.percentile(img, [plow, phigh])
    if hi - lo < 1e-9:
        return np.zeros_like(img)
    scale = target_hi / (hi - lo)
    return np.clip((img - lo) * scale, 0.0, 1.0).astype(np.float32)


# ── Channel alignment ──────────────────────────────────────────────────────────

_ALIGN_SEARCH_PX = 25.0    # search radius when max_shift_px is 0 (no clamp)
_ALIGN_MIN_NCC = 0.5       # real pairs measure 0.89–1.00; below this = no common structure


def _alignment_signal(img: np.ndarray) -> np.ndarray:
    """Gradient magnitude of *img* with the sky floored to zero.

    Edges (limb, ring, belts) are what must coincide for a fringe-free
    composite, and their position does not depend on each filter's
    brightness profile the way raw intensity does — Saturn's rings and IR
    vs. visible limb darkening differ strongly in brightness between
    filters. The floor (8% of the bright end) zeroes the stacked frames'
    sky plateau, whose hard rectangular coverage edge differs per filter
    and would otherwise dominate the gradient.
    """
    a = img.astype(np.float32)
    hi = float(np.percentile(a, 99.9))
    t = 0.08 * hi
    a = np.clip((a - t) / max(hi - t, 1e-9), 0.0, None)
    a = cv2.GaussianBlur(a, (0, 0), 1.5)
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1)
    return np.sqrt(gx * gx + gy * gy)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def _parabolic_offset(lo: float, mid: float, hi: float) -> float:
    d = lo - 2.0 * mid + hi
    return 0.0 if abs(d) < 1e-12 else 0.5 * (lo - hi) / d


def estimate_channel_shift(
    ref: np.ndarray,
    img: np.ndarray,
    search_px: float = _ALIGN_SEARCH_PX,
) -> Optional[Tuple[float, float, float]]:
    """Shift (dx, dy) to apply to *img* (via apply_shift) to register it on
    *ref*, plus the peak NCC. None when no trustworthy peak exists.

    Whole-frame normalised cross-correlation of _alignment_signal() over
    ±search_px (no disk-ROI crop — the user has explicitly rejected that
    approach twice; do not reintroduce it in any form, gated or not).
    Integer peak from cv2.matchTemplate, then refined on a 0.25 px grid
    (synthetic known-shift test on real Saturn data: rms 0.04 px, max
    0.08 px; parabolic interpolation alone left 0.3 px errors along the
    rings' long axis).

    Replaces cv2.phaseCorrelate: its spectral whitening amplifies the
    high-frequency noise of weak channels, and on real data it lost the
    peak entirely (response ≈ 0, shifts of 150 px) or locked onto plausible
    but wrong offsets (Jupiter B in ~25% of windows; Saturn G/B up to 5 px
    off). Gradient NCC found a sane peak (NCC 0.89–1.00) for every pair of
    a 27-window Saturn and a 30-window Jupiter session, and composites
    built with it were visibly fringe-free where phase correlation's were
    not (D:/Astro/Sat/271026 w06/w07/w22, Mono_Temp_260407 w06/w26).

    Returns None when the peak lies on the search boundary (true offset
    outside the range) or its NCC is below _ALIGN_MIN_NCC.
    """
    S = max(1, int(np.ceil(search_px)))
    gr = _alignment_signal(ref)
    gi = _alignment_signal(img)
    padded = cv2.copyMakeBorder(gr, S, S, S, S, cv2.BORDER_CONSTANT, value=0)
    m = cv2.matchTemplate(padded, gi, cv2.TM_CCOEFF_NORMED)
    i, j = np.unravel_index(int(np.argmax(m)), m.shape)
    peak = float(m[i, j])
    if i in (0, 2 * S) or j in (0, 2 * S) or peak < _ALIGN_MIN_NCC:
        return None

    dx = j - S + _parabolic_offset(m[i, j - 1], m[i, j], m[i, j + 1])
    dy = i - S + _parabolic_offset(m[i - 1, j], m[i, j], m[i + 1, j])
    offs = np.arange(-2, 3) * 0.25
    for _ in range(2):
        sx = [_ncc(gr, apply_shift(gi, dx + o, dy)) for o in offs]
        k = int(np.argmax(sx))
        dx += offs[k] + (0.25 * _parabolic_offset(sx[k - 1], sx[k], sx[k + 1]) if 0 < k < 4 else 0.0)
        sy = [_ncc(gr, apply_shift(gi, dx, dy + o)) for o in offs]
        k = int(np.argmax(sy))
        dy += offs[k] + (0.25 * _parabolic_offset(sy[k - 1], sy[k], sy[k + 1]) if 0 < k < 4 else 0.0)
    if abs(dx) > search_px or abs(dy) > search_px:
        return None
    return float(dx), float(dy), peak


def align_channels(
    channels: Dict[str, np.ndarray],
    reference_key: str,
    max_shift_px: float = 0.0,
    estimate_from: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, float]]]:
    """Align all channels to *reference_key* (see estimate_channel_shift).

    Args:
        channels:      {filter_name: float [0,1] 2D image}
        reference_key: Key of the channel to treat as reference (no shift applied).
        max_shift_px:  Search radius; a true offset beyond it is rejected
                       (0 = default radius _ALIGN_SEARCH_PX).
        estimate_from: Optional images (same keys and shapes as *channels*)
                       to measure the shifts on, which are then applied to
                       *channels*. Used to measure on the unsharpened step04
                       stacks: wavelet sharpening amplifies each filter's
                       own noise and PSF, which degrades the match (Saturn
                       271026: peak NCC 0.79–0.98 sharpened vs. 0.93–0.99
                       unsharpened, and visibly worse composites).

    Returns:
        (aligned, shifts):
            aligned: New dict with aligned images (reference unchanged, others shifted).
            shifts:  {filter_name: (dx, dy)} actually applied for every key
                     (0.0, 0.0 for the reference and for any rejected estimate).
    """
    src = estimate_from if estimate_from is not None else channels
    shifts = _estimate_shifts(src, reference_key, list(channels), max_shift_px)
    return _apply_shifts(channels, shifts), shifts


def _estimate_shifts(
    src: Dict[str, np.ndarray],
    reference_key: str,
    keys,
    max_shift_px: float,
) -> Dict[str, Tuple[float, float]]:
    search = max_shift_px if max_shift_px > 0 else _ALIGN_SEARCH_PX
    shifts: Dict[str, Tuple[float, float]] = {}
    for key in keys:
        est = None if key == reference_key else estimate_channel_shift(src[reference_key], src[key], search)
        shifts[key] = (est[0], est[1]) if est is not None else (0.0, 0.0)
    return shifts


def _apply_shifts(
    channels: Dict[str, np.ndarray],
    shifts: Dict[str, Tuple[float, float]],
) -> Dict[str, np.ndarray]:
    return {
        k: (apply_shift(img, *shifts[k]) if shifts.get(k, (0.0, 0.0)) != (0.0, 0.0) else img)
        for k, img in channels.items()
    }


def _group_signal(images: Dict[str, np.ndarray], keys, shifts: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """Mean of the brightness-normalised, already-registered *keys* images —
    the colour group's combined appearance, for aligning it to L as a unit."""
    acc = None
    for k in keys:
        a = images[k].astype(np.float32)
        a = a / max(float(np.percentile(a, 99.9)), 1e-9)
        if shifts.get(k, (0.0, 0.0)) != (0.0, 0.0):
            a = apply_shift(a, *shifts[k])
        acc = a if acc is None else acc + a
    return acc / len(keys)


def align_lrgb(
    channels: Dict[str, np.ndarray],
    lum_key: str,
    colour_keys,
    colour_ref: str,
    max_shift_px: float = 0.0,
    estimate_from: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, float]]]:
    """Two-stage LRGB alignment: colour channels to *colour_ref* first, then
    the registered colour group as a whole to the luminance channel.

    Registering each colour channel to L independently lets every one of
    them carry its own IR-vs-visible matching error, so the colour channels
    end up misregistered against EACH OTHER — the visible fringe. Here the
    colour channels are matched among themselves (similar appearance), and
    the group then moves as one rigid unit, so any residual L-vs-colour
    error only offsets colour from luminance instead of splitting colours.
    Each channel is still interpolated once, with the summed shift.
    """
    src = estimate_from if estimate_from is not None else channels
    shifts = _estimate_shifts(src, colour_ref, colour_keys, max_shift_px)

    search = max_shift_px if max_shift_px > 0 else _ALIGN_SEARCH_PX
    est = estimate_channel_shift(src[lum_key], _group_signal(src, colour_keys, shifts), search)
    gdx, gdy = (est[0], est[1]) if est is not None else (0.0, 0.0)
    shifts = {k: (dx + gdx, dy + gdy) for k, (dx, dy) in shifts.items()}
    shifts[lum_key] = (0.0, 0.0)
    for k in channels:
        shifts.setdefault(k, (0.0, 0.0))
    return _apply_shifts(channels, shifts), shifts


# ── RGB composite ──────────────────────────────────────────────────────────────

def make_rgb(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Stack R, G, B channels into an (H, W, 3) float [0, 1] RGB image."""
    return np.stack([r, g, b], axis=2).astype(np.float32)


# ── LRGB composite ─────────────────────────────────────────────────────────────

def make_lrgb(
    luminance: np.ndarray,
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    lrgb_weight: float = 1.0,
) -> np.ndarray:
    """LRGB composite in Lab colour space.

    Blends the luminance channel (e.g. IR) into the L channel of the RGB image.

    Args:
        luminance:    2-D float [0, 1] luminance image (e.g. IR filter).
        r, g, b:      2-D float [0, 1] colour channels.
        lrgb_weight:  Weight of the external luminance vs. the RGB's own L.
                      1.0 = fully replace with luminance, 0.0 = keep RGB L.

    Returns:
        (H, W, 3) float [0, 1] RGB image.
    """
    rgb = np.stack([r, g, b], axis=2).astype(np.float32)

    # Convert RGB → Lab  (cv2 float32 input expects [0,1]; L output is [0,100])
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)

    lum_100 = luminance.astype(np.float32) * 100.0
    lab[:, :, 0] = (lrgb_weight * lum_100
                    + (1.0 - lrgb_weight) * lab[:, :, 0])

    result = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


# ── High-level compose ─────────────────────────────────────────────────────────

def compose(
    spec,  # CompositeSpec (avoid circular import with config)
    filter_images: Dict[str, np.ndarray],
    align: bool = True,
    max_shift_px: float = 0.0,
    color_stretch_mode: str = "joint",
    stretch_plow: float = 0.1,
    stretch_phigh: float = 99.9,
    stretch_target_hi: float = 1.0,
    saturate: bool = False,
    saturation_phigh: float = 99.5,
    saturation_headroom: float = 0.30,
    align_images: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, dict]:
    """Build a composite image from per-filter images according to *spec*.

    Args:
        spec:               CompositeSpec defining the channel mapping.
        filter_images:      {filter_name: float [0,1] 2D array}
        align:              If True, align channels before compositing.
        max_shift_px:       Max allowed alignment shift (0 = no clamp).
        align_images:       Optional {filter_name: 2D array} of the same
                            shape to measure alignment shifts on (the
                            unsharpened step04 stacks); falls back to
                            *filter_images* when None or incomplete.
        color_stretch_mode: How to stretch colour channels (R/G/B):
                              "joint"       – same lo/hi from all colour channels
                                              combined (preserves colour ratios)
                              "independent" – each channel independently
                              "none"        – no pre-stretch (native values)
        stretch_plow:       Lower percentile (joint/independent modes).
        stretch_phigh:      Upper percentile (joint/independent modes).

    Returns:
        (composite_image, log_dict)
        composite_image: (H, W, 3) float [0, 1]
        log_dict:        stretch and alignment details per channel
    """
    required = {spec.R, spec.G, spec.B}
    lum_key = spec.L
    if lum_key is not None:
        required.add(lum_key)

    missing = required - set(filter_images.keys())
    if missing:
        raise ValueError(f"Missing filters for composite '{spec.name}': {missing}")

    colour_keys = {spec.R, spec.G, spec.B}
    stretched: Dict[str, np.ndarray] = {}
    stretch_log: Dict[str, dict] = {}

    # ── Luminance channel ──────────────────────────────────────────────────────
    # Mirror the colour stretch mode: if colour channels are not stretched
    # (mode="none"), luminance must also stay at native intensity so that both
    # operate on the same scale before Lab L replacement.  Mismatch (e.g.
    # colour native max≈0.70 while luminance stretched to 1.0) inflates Lab L
    # from ~71 to 100, making the LRGB composite visibly too bright.
    if lum_key is not None:
        img = filter_images[lum_key]
        if color_stretch_mode == "none":
            stretched[lum_key] = img.astype(np.float32)
            stretch_log[lum_key] = {"mode": "none"}
        else:
            lo = float(np.percentile(img, stretch_plow))
            hi = float(np.percentile(img, stretch_phigh))
            stretched[lum_key] = auto_stretch(img, stretch_plow, stretch_phigh, stretch_target_hi)
            stretch_log[lum_key] = {"mode": "independent",
                                    "plow": round(lo, 5), "phigh": round(hi, 5)}

    # ── Colour channels ────────────────────────────────────────────────────────
    if color_stretch_mode == "joint":
        # Single lo/hi from all colour channels combined → preserves colour ratios
        combined = np.concatenate([filter_images[k].ravel() for k in colour_keys])
        lo = float(np.percentile(combined, stretch_plow))
        hi = float(np.percentile(combined, stretch_phigh))
        span = hi - lo if hi > lo else 1.0
        scale = stretch_target_hi / span
        for key in colour_keys:
            stretched[key] = np.clip(
                (filter_images[key] - lo) * scale, 0.0, 1.0
            ).astype(np.float32)
            stretch_log[key] = {"mode": "joint",
                                "plow": round(lo, 5), "phigh": round(hi, 5)}
    elif color_stretch_mode == "independent":
        for key in colour_keys:
            img = filter_images[key]
            lo = float(np.percentile(img, stretch_plow))
            hi = float(np.percentile(img, stretch_phigh))
            stretched[key] = auto_stretch(img, stretch_plow, stretch_phigh, stretch_target_hi)
            stretch_log[key] = {"mode": "independent",
                                "plow": round(lo, 5), "phigh": round(hi, 5)}
    else:  # "none"
        for key in colour_keys:
            stretched[key] = filter_images[key].astype(np.float32)
            stretch_log[key] = {"mode": "none"}

    # ── Alignment reference ────────────────────────────────────────────────────
    # Use a stable, fixed reference to prevent frame-to-frame composite jitter.
    # Dynamic selection (max by 95th percentile) varies per frame when channels
    # have different brightness, causing the composite planet position to shift.
    if getattr(spec, "align_ref", None) is not None:
        reference_key = spec.align_ref
    elif lum_key is not None:
        reference_key = lum_key
    else:
        # Prefer IR (best seeing quality) → R → first available colour channel
        _ALIGN_PREF = ["IR", "R", "G", "B", "CH4"]
        reference_key = next(
            (k for k in _ALIGN_PREF if k in required),
            spec.R,
        )

    # ── Align ──────────────────────────────────────────────────────────────────
    shift_log: Dict[str, list] = {k: [0.0, 0.0] for k in required}
    align_source = None
    if align and len(stretched) > 1:
        est = None
        if align_images is not None and all(
            k in align_images and align_images[k].shape == stretched[k].shape for k in stretched
        ):
            est = align_images
        align_source = "align_images" if est is not None else "filter_images"
        if lum_key is not None and reference_key == lum_key and len(colour_keys) > 1:
            colour_ref = next((k for k in ["R", "G", "B", "IR", "CH4"] if k in colour_keys), spec.R)
            aligned, shifts = align_lrgb(stretched, lum_key, colour_keys, colour_ref,
                                         max_shift_px=max_shift_px, estimate_from=est)
        else:
            aligned, shifts = align_channels(stretched, reference_key,
                                             max_shift_px=max_shift_px, estimate_from=est)
        for key, (dx, dy) in shifts.items():
            shift_log[key] = [round(dx, 3), round(dy, 3)]
    else:
        aligned = stretched

    # NOTE: Pre-channel per-channel brightness masking was removed.
    # Blending each channel toward its sky background at a fixed radius creates
    # an artificially steep brightness drop at the limb (≈2× steeper than natural
    # limb darkening), producing a visible circular boundary in the composite.
    # Colour-fringe suppression is handled instead by the post-composite Lab
    # desaturation below, which affects only chrominance (a, b), not luminance.

    # ── Compose ────────────────────────────────────────────────────────────────
    r_img, g_img, b_img = aligned[spec.R], aligned[spec.G], aligned[spec.B]

    if lum_key is not None:
        result = make_lrgb(aligned[lum_key], r_img, g_img, b_img,
                           lrgb_weight=spec.lrgb_weight)
    else:
        result = make_rgb(r_img, g_img, b_img)

    # ── Post-composite limb desaturation — DISABLED 2026-08-09 (see below) ─────
    # ── Post-composite limb desaturation ───────────────────────────────────────
    # The soft pre-channel mask corrects the outer limb zone (r > r_ref), but the
    # inner limb colour fringe (r ≈ 0.92–1.0 × r_ref) remains where mask≈1.0.
    # Root cause: wavelength-dependent limb darkening makes G's disk appear larger
    # than B by ~1.5 px, creating a thin colour zone at the edge.
    # Fix: after compositing, apply a Lab-space saturation taper in the limb zone.
    # This only reduces colour (a, b channels), leaving luminance (L) untouched,
    # so the natural limb-darkening gradient is preserved.
    #
    # -------------------------------------------------------------------------
    # WHY THIS IS DISABLED (2026-08-09 Saturn investigation):
    #
    # The band above was tuned to desat_start=0.89×r_ref, desat_width=0.15×r_ref
    # — i.e. it forces a/b to ~0 across the OUTER 15% OF THE DISK RADIUS. That
    # is roughly 10px wide on this session's ~65px-radius Saturn disk, applied
    # to fix a stated root cause of only ~1.5px. That mismatch (a band ~7x
    # wider than the physical effect it claims to correct) strongly suggests
    # it was actually tuned empirically against a much bigger bug that existed
    # at the time: composite.py's channel-to-channel alignment (align_channels/
    # subpixel_align) used to run cv2.phaseCorrelate() on the WHOLE frame,
    # including Saturn's ring — a large, per-filter-brightness-inconsistent
    # feature that could bias the correlation peak. Measured real misalignment
    # from that bug was up to 14.6px (Saturn_Data/step06_rgb_composite/
    # window_04, B channel) — nowhere near the "~1.5px" this desaturation
    # docstring cites, but roughly consistent with needing a ~10px-wide mask
    # to hide it after the fact.
    #
    # The actual root cause is believed to be composite.py's channel-to-channel
    # alignment (align_channels/subpixel_align) running cv2.phaseCorrelate() on
    # the WHOLE frame, including Saturn's ring — a large, per-filter-brightness-
    # inconsistent feature that can bias the correlation peak. Measured real
    # misalignment from that bug was up to 14.6px (Saturn_Data/step06_rgb_composite/
    # window_04, B channel) — far larger than the "~1.5px" effect this
    # desaturation was designed for, and roughly consistent with needing a
    # ~10px-wide mask to hide it after the fact.
    #
    # A proper fix for that misalignment (disk-ROI-cropped channel alignment,
    # plus sharing one filter's disk geometry across all filters during
    # de-rotation so they don't each drift to a slightly different centre) is
    # still being developed and verified separately — NOT included in this
    # commit. This desaturation block is disabled now because, even before that
    # fix lands, it is already doing more harm than good on real Saturn data:
    # visually it produced a small, vividly-coloured inner sphere sitting
    # inside a much larger grey sphere, wiping out real colour across the
    # outer 15% of the disk radius (visually confirmed:
    # Saturn_Data/step06_compare/desat_before_after.png).
    #
    # RISK / WHY THIS IS COMMENTED OUT RATHER THAN DELETED:
    # This function is shared by every composite spec and every target
    # (Jupiter/Mars/Venus too), not just Saturn. The ~1.5px wavelength-
    # dependent limb-darkening effect this was originally meant to catch is a
    # real physical phenomenon independent of the alignment bug above, and it
    # has NOT been re-verified against real Jupiter/Mars data. If a thin
    # (~1-2px) colour fringe reappears at the limb on non-Saturn data:
    #   - Do NOT just re-enable this block as-is — first confirm with a
    #     chroma-vs-radius measurement whether a real registration bug is at
    #     play, since that should be fixed at the source, not masked.
    #   - If a genuine small residual fringe is confirmed, re-enable with a
    #     MUCH narrower band matching the originally-stated ~1.5px effect
    #     (e.g. desat_start≈0.97×r_ref, desat_width≈0.05×r_ref) rather than
    #     the current 0.89/0.15 values, which were sized for the old bug.
    #
    # try:
    #     ref_img_d = aligned[reference_key]
    #     cx_d, cy_d, r_d, _, _ = find_disk_center(ref_img_d)
    #     h_d, w_d = ref_img_d.shape[:2]
    #     yy_d, xx_d = np.ogrid[:h_d, :w_d]
    #     dist_d = np.sqrt((xx_d - cx_d) ** 2 + (yy_d - cy_d) ** 2).astype(np.float32)
    #     # Start desaturation at 0.89×r_ref to catch the inner limb colour fringe.
    #     # At r=0.89×r: mask=1 (no effect); at r=0.93×r: mask≈0.72 (28% desat);
    #     # at r=0.97×r: mask≈0.30 (70% desat).  Belt features end at ~0.86×r so
    #     # the equatorial colour region is barely touched (<7% at r=0.9×r).
    #     desat_start = r_d * 0.89
    #     desat_width = r_d * 0.15          # fade completes at ~1.04×r_ref
    #     t_d = np.clip((dist_d - desat_start) / desat_width, 0.0, 1.0)
    #     desat_mask = (0.5 * (1.0 + np.cos(np.pi * t_d))).astype(np.float32)
    #     # Convert composite → Lab, suppress a/b, convert back
    #     lab_r = cv2.cvtColor(result, cv2.COLOR_RGB2Lab)
    #     lab_r[:, :, 1] *= desat_mask   # a channel
    #     lab_r[:, :, 2] *= desat_mask   # b channel
    #     result = np.clip(cv2.cvtColor(lab_r, cv2.COLOR_Lab2RGB), 0.0, 1.0).astype(np.float32)
    # except Exception:
    #     pass  # disk detection failed — skip post-composite desaturation

    # ── Auto saturation boost ──────────────────────────────────────────────────
    sat_gain: Optional[float] = None
    if saturate:
        try:
            ref_img_s = aligned[reference_key]
            cx_s, cy_s, r_s, _, _ = find_disk_center(ref_img_s)
            h_s, w_s = ref_img_s.shape[:2]
            yy_s, xx_s = np.ogrid[:h_s, :w_s]
            # Sample inside 0.85× radius to stay away from limb fringe
            disk_mask = ((xx_s - cx_s) ** 2 + (yy_s - cy_s) ** 2) <= (r_s * 0.85) ** 2

            lab_s = cv2.cvtColor(result, cv2.COLOR_RGB2Lab)
            current = float(np.percentile(
                np.sqrt(lab_s[:, :, 1] ** 2 + lab_s[:, :, 2] ** 2)[disk_mask],
                saturation_phigh,
            ))
            if current >= 1e-3:
                sat_gain = float(np.clip(127.0 * saturation_headroom / current, 1.0, 2.0))
                lab_s[:, :, 1] = np.clip(lab_s[:, :, 1] * sat_gain, -127.0, 127.0)
                lab_s[:, :, 2] = np.clip(lab_s[:, :, 2] * sat_gain, -127.0, 127.0)
                result = np.clip(cv2.cvtColor(lab_s, cv2.COLOR_Lab2RGB), 0.0, 1.0).astype(np.float32)
        except Exception:
            pass  # disk detection failed — skip saturation boost

    log = {
        "type":               "LRGB" if lum_key else "RGB",
        "color_stretch_mode": color_stretch_mode,
        "channels":           {"L": lum_key, "R": spec.R, "G": spec.G, "B": spec.B},
        "stretch":            stretch_log,
        "alignment":          shift_log,
        "alignment_source":   align_source,
        "saturation_gain":    round(sat_gain, 4) if sat_gain is not None else None,
    }
    return result, log
