"""CV-based satellite/shadow position refinement (opt-in, per-window, confidence-gated).

This module promotes the box-mult-sweep CV re-detection method used
throughout the shadow/satellite rotational-bias investigation
(docs/investigation_shadow_rotational_bias.md) as a DIAGNOSTIC tool into a
narrowly-scoped PRODUCTION refinement: given a window's image and the
ephemeris-predicted pixel position of an on-disk moon or shadow, attempt to
re-detect it with CV and return the resulting pixel offset -- but only when
that re-detection clears a 3-sigma confidence bar (stability across box
sizes + peak significance) AND the resulting offset is not implausibly large.

Two object types, two very different detectability profiles:
  - "satellite" (moon body): only Jupiter's CH4 (methane-band) filter shows
    reliable native contrast -- methane absorption darkens the disk, not the
    moons, so IR/R/G/B show essentially no detectable satellite signal. A
    caller must measure this on CH4 specifically and then broadcast the
    resulting raw-pixel offset to other filters' own predictions (see
    pipeline/steps/satellite_composite.py) -- this module does not do that
    broadcast itself.
  - "shadow": real signal (a dark spot) in every filter, since a shadow is
    a literal absence of sunlight. No cross-filter anchoring is needed --
    each filter can refine its own shadow prediction against its own image.

This is intentionally NOT the same thing as the global CV-based calibration
pattern this project rejects everywhere else (fitting plate_scale, pole_pa,
or a disk-center offset against CV positions and reusing that fit across
windows/sessions). The offset computed here is local to one window (and, for
shadows, one filter), applied once, gated on its own confidence, and never
cached or reused elsewhere -- a failed or low-confidence case silently falls
back to pure ephemeris. See pipeline/config.py's
SatelliteConfig.cv_position_refinement_enabled and
docs/investigation_shadow_rotational_bias.md for the full rationale.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
from scipy.ndimage import gaussian_filter

from pipeline.modules.satellite_tracker import SatellitePos

_BOX_MULTS = (2.0, 3.0, 4.0, 5.0)

ObjType = Literal["satellite", "shadow"]


@dataclass
class RefinementResult:
    """A confidence-gated CV re-detection offset for one object in one window."""
    dx_px: float
    dy_px: float
    confidence: str    # "high" | "medium" -- "low" never reaches this dataclass
    peak_sig: float
    stability_px: Optional[float]


def _robust_redetect(img: np.ndarray, x_px: float, y_px: float, r_px: float, obj_type: ObjType) -> List[dict]:
    """Box-mult-sweep (2/3/4/5x r_px) weighted-centroid re-detection.

    obj_type="satellite" looks for a positive (brighter-than-background)
    signal; obj_type="shadow" looks for a negative (darker-than-background)
    one. Identical algorithm to the one used throughout the investigation
    (experiments/full_cv_verified_contrast.py's robust_redetect), reproduced
    here as tested production code rather than an ad-hoc script.
    """
    h, w = img.shape
    per_box = []
    for box_mult in _BOX_MULTS:
        pad = int(math.ceil(box_mult * r_px))
        x0, x1 = max(0, int(round(x_px)) - pad), min(w, int(round(x_px)) + pad)
        y0, y1 = max(0, int(round(y_px)) - pad), min(h, int(round(y_px)) + pad)
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            per_box.append({"box_mult": box_mult, "centroid": None, "peak_sig": None})
            continue
        smoothed = gaussian_filter(crop, sigma=1.0)
        ys, xs = np.mgrid[y0:y1, x0:x1]
        dist = np.sqrt((xs - x_px) ** 2 + (ys - y_px) ** 2)
        annulus = (dist >= 1.5 * r_px) & (dist <= 2.5 * r_px)
        if annulus.sum() < 5:
            per_box.append({"box_mult": box_mult, "centroid": None, "peak_sig": None})
            continue
        bg_vals = smoothed[annulus]
        bg_median = float(np.median(bg_vals))
        sigma_bg = float(np.std(bg_vals))
        if sigma_bg <= 1e-9:
            per_box.append({"box_mult": box_mult, "centroid": None, "peak_sig": None})
            continue
        sig = (bg_median - smoothed) / sigma_bg if obj_type == "shadow" else (smoothed - bg_median) / sigma_bg
        within_r = dist <= (1.5 * r_px)
        peak_sig = float(sig[within_r].max()) if within_r.any() else None
        candidate = (sig >= 3.0) & within_r
        if not candidate.any():
            per_box.append({"box_mult": box_mult, "centroid": None, "peak_sig": peak_sig})
            continue
        w_sig = sig[candidate]
        cx = float(np.sum(xs[candidate] * w_sig) / np.sum(w_sig))
        cy = float(np.sum(ys[candidate] * w_sig) / np.sum(w_sig))
        per_box.append({"box_mult": box_mult, "centroid": (cx, cy), "peak_sig": peak_sig})
    return per_box


def _pairwise_stability(per_box: List[dict]) -> Optional[float]:
    pts = [b["centroid"] for b in per_box if b["centroid"] is not None]
    if len(pts) < 2:
        return None
    return max(
        math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
        for i in range(len(pts)) for j in range(i + 1, len(pts))
    )


def _assign_confidence(stability_px: Optional[float], peak_sig_box3: Optional[float]) -> str:
    if stability_px is not None and stability_px <= 2.0 and peak_sig_box3 is not None and peak_sig_box3 >= 4.0:
        return "high"
    if stability_px is not None and stability_px <= 5.0 and peak_sig_box3 is not None and peak_sig_box3 >= 3.0:
        return "medium"
    return "low"


def _recommended_position(per_box: List[dict]):
    by_mult = {b["box_mult"]: b for b in per_box}
    for m in (3.0, 2.0):
        b = by_mult.get(m)
        if b is not None and b["centroid"] is not None:
            return b["centroid"]
    return None


def refine_position_via_cv(
    image: np.ndarray,
    x_pred: float,
    y_pred: float,
    r_px: float,
    obj_type: ObjType,
    max_offset_px: float = 20.0,
) -> Optional[RefinementResult]:
    """Attempt a confidence-gated CV re-detection of a satellite body or shadow.

    Returns None (caller must fall back to pure ephemeris) when: no signal
    clears the 3-sigma detection threshold, confidence is "low", or the
    resulting offset magnitude exceeds max_offset_px -- the latter guards
    against the CV re-detection method's own known systematic bias from real
    belt/zone brightness gradients (docs/investigation_shadow_rotational_bias.md
    section 12-11): that bias is reproducible (not noisy), so it can present
    as high confidence while still being wrong. An offset larger than the
    largest bias magnitude ever confirmed in this investigation is treated
    as a mis-lock, not a correction.
    """
    per_box = _robust_redetect(image, x_pred, y_pred, r_px, obj_type)
    rec = _recommended_position(per_box)
    if rec is None:
        return None
    stability = _pairwise_stability(per_box)
    by_mult = {b["box_mult"]: b for b in per_box}
    peak_sig3 = by_mult.get(3.0, {}).get("peak_sig")
    confidence = _assign_confidence(stability, peak_sig3)
    if confidence == "low":
        return None
    dx, dy = rec[0] - x_pred, rec[1] - y_pred
    if math.hypot(dx, dy) > max_offset_px:
        return None
    return RefinementResult(
        dx_px=dx, dy_px=dy, confidence=confidence,
        peak_sig=float(peak_sig3), stability_px=stability,
    )


def apply_offset(
    positions: List[Optional[SatellitePos]],
    dx: float,
    dy: float,
    disk_cx: float,
    disk_cy: float,
    disk_r: float,
) -> List[Optional[SatellitePos]]:
    """Shift a list of SatellitePos by (dx, dy), recomputing dist_px/on_disk
    relative to the given disk. Logs (prints) whenever on_disk flips, since a
    limb-adjacent object can cross the disk boundary as a side effect of the
    shift -- this changes whether/how it gets composited downstream."""
    out: List[Optional[SatellitePos]] = []
    for p in positions:
        if p is None:
            out.append(None)
            continue
        nx, ny = p.x_px + dx, p.y_px + dy
        dist = math.hypot(nx - disk_cx, ny - disk_cy)
        new_on_disk = dist < disk_r
        if new_on_disk != p.on_disk:
            print(f"    [cv_refinement] on_disk flip for {p.name}: {p.on_disk} -> {new_on_disk}")
        out.append(dataclasses.replace(p, x_px=nx, y_px=ny, dist_px=dist, on_disk=new_on_disk))
    return out
