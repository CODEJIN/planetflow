"""Galilean satellite position tracker for Jupiter transit detection.

Queries JPL Horizons for Io/Europa/Ganymede/Callisto positions relative to
Jupiter's disk center and converts them to pixel coordinates.  When Horizons
is unavailable, falls back to OpenCV blob detection in the image data.

Satellite Horizons IDs:
    Io=501, Europa=502, Ganymede=503, Callisto=504
"""
from __future__ import annotations

import re
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Horizons IDs for the four Galilean moons
GALILEAN_MOONS: Dict[str, str] = {
    "Io":       "501",
    "Europa":   "502",
    "Ganymede": "503",
    "Callisto": "504",
}

# Jupiter equatorial radius (km) — for fallback plate-scale estimation
_JUP_EQ_RADIUS_KM = 71_492.0
# Jupiter polar radius (km) — oblate spheroid, flattening ~6.5%
_JUP_POL_RADIUS_KM = 66_854.0
# Typical Jupiter angular radius (arcsec) when ~5.5-6 AU away from Earth (2026)
_JUP_ANG_RADIUS_FALLBACK_ARCSEC = 18.0   # ~36 arcsec diameter

# Jupiter north pole direction in ICRF (IAU 2009, approximately constant over decades)
# RA=268.057°, Dec=64.495°
_JUP_POLE_ICRF = np.array([
    np.cos(np.radians(64.495)) * np.cos(np.radians(268.057)),
    np.cos(np.radians(64.495)) * np.sin(np.radians(268.057)),
    np.sin(np.radians(64.495)),
], dtype=np.float64)

# Skyfield ephemeris kernels for 3D shadow position computation
# Resolution order: env PLANETFLOW_SKYFIELD_DIR → ~/.planetflow/skyfield → /tmp/skyfield
import os as _os
_SKYFIELD_KERNEL_DIR = Path(
    _os.environ.get("PLANETFLOW_SKYFIELD_DIR", "")
    or Path.home() / ".planetflow" / "skyfield"
)
_SKYFIELD_PLANETS_BSP = "de440s.bsp"
_SKYFIELD_MOONS_BSP   = "jup365.bsp"
_BSP_URLS = {
    "de440s.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp",
    "jup365.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/jup365.bsp",
}
_MOON_SF_ID: Dict[str, str] = {
    "Io":       "io",
    "Europa":   "europa",
    "Ganymede": "ganymede",
    "Callisto": "callisto",
}


@dataclass
class SatellitePos:
    """Pixel-space position of a Galilean satellite at one timestamp."""
    name:     str
    x_px:     float    # pixel column (positive = right / East if flip_ew=False)
    y_px:     float    # pixel row    (positive = down  / South)
    on_disk:  bool     # True if sqrt((x-cx)²+(y-cy)²) < disk_radius_px
    dist_px:  float    # distance from disk center in pixels

    def offset_px(self, disk_cx: float, disk_cy: float) -> Tuple[float, float]:
        """Return (dx, dy) from disk center."""
        return self.x_px - disk_cx, self.y_px - disk_cy


class SatelliteTracker:
    """Query and cache Galilean satellite pixel positions.

    Args:
        jupiter_horizons_id: Horizons target ID for Jupiter (default "599").
        observer_code:       JPL Horizons observer center (default geocentric).
        flip_ew:             Mirror East-West (True = East on left, astronomical
                             convention).  False = East on right (default for
                             many planetary cameras with alt-az mounts).
        flip_ns:             False = North-up camera (north = −y, default).
                             True  = South-up camera (south = −y).
    """

    def __init__(
        self,
        jupiter_horizons_id: str = "599",
        observer_code: str = "500@399",
        flip_ew: bool = False,
        flip_ns: bool = False,
    ) -> None:
        self.jupiter_id    = jupiter_horizons_id
        self.observer_code = observer_code
        self.flip_ew       = flip_ew
        self.flip_ns       = flip_ns
        # Cache: command_str → [(datetime, ra_deg, dec_deg)]
        self._ra_dec_cache: Dict[str, List[Tuple[datetime, float, float]]] = {}
        self._plate_scale: Optional[float] = None   # arcsec/px (computed once)

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_plate_scale(
        self,
        disk_radius_px: float,
        t_sample: datetime,
    ) -> float:
        """Return plate scale in arcsec/px from disk radius and Horizons ang-diam."""
        if self._plate_scale is not None:
            return self._plate_scale

        ang_radius = self._query_angular_radius_arcsec(t_sample)
        if ang_radius is None:
            ang_radius = _JUP_ANG_RADIUS_FALLBACK_ARCSEC
            warnings.warn(
                f"[SatTracker] Horizons ang-diam failed → using fallback "
                f"{_JUP_ANG_RADIUS_FALLBACK_ARCSEC:.1f} arcsec as Jupiter angular radius"
            )

        plate_scale = ang_radius / max(disk_radius_px, 1.0)
        print(
            f"  [SatTracker] plate scale = {plate_scale:.5f} arcsec/px  "
            f"(ang_radius={ang_radius:.2f}\", disk_r={disk_radius_px:.1f}px)"
        )
        self._plate_scale = plate_scale
        return plate_scale

    def get_positions(
        self,
        t_list: List[datetime],
        disk_cx: float,
        disk_cy: float,
        disk_radius_px: float,
        plate_scale_arcsec_per_px: Optional[float] = None,
        pole_pa_deg: float = 0.0,
        np_ang_deg: float = 0.0,
    ) -> Dict[str, List[SatellitePos]]:
        """Return pixel positions for all four Galilean moons at each timestamp.

        Args:
            t_list:                    List of UTC datetimes (one per frame).
            disk_cx, disk_cy:          Disk center in pixels (from find_disk_center).
            disk_radius_px:            Disk semi-major radius in pixels.
            plate_scale_arcsec_per_px: If None, queried from Horizons.

        Returns:
            {moon_name: [SatellitePos, ...]}  — same length as t_list.
            Moons for which Horizons data is unavailable are omitted.
        """
        if not t_list:
            return {}

        # Strip timezone info so comparisons with Horizons naive datetimes work
        t_list_naive = [t.replace(tzinfo=None) if t.tzinfo is not None else t for t in t_list]

        if plate_scale_arcsec_per_px is None:
            plate_scale_arcsec_per_px = self.get_plate_scale(disk_radius_px, t_list_naive[0])

        t_start = min(t_list_naive) - timedelta(minutes=5)
        t_end   = max(t_list_naive) + timedelta(minutes=5)

        jup_ephem = self._query_ra_dec(self.jupiter_id, t_start, t_end)
        if not jup_ephem:
            warnings.warn("[SatTracker] Jupiter ephemeris unavailable → no satellite positions")
            return {}

        results: Dict[str, List[SatellitePos]] = {}
        for moon_name, moon_id in GALILEAN_MOONS.items():
            moon_ephem = self._query_ra_dec(moon_id, t_start, t_end)
            if not moon_ephem:
                warnings.warn(f"[SatTracker] {moon_name} ephemeris unavailable → skipped")
                continue

            positions: List[SatellitePos] = []
            for t in t_list_naive:
                jup_ra, jup_dec = _interp_ra_dec(jup_ephem, t)
                moon_ra, moon_dec = _interp_ra_dec(moon_ephem, t)

                # Angular offset in arcsec
                dra_arcsec  = (moon_ra  - jup_ra)  * np.cos(np.radians(jup_dec)) * 3600.0
                ddec_arcsec = (moon_dec - jup_dec) * 3600.0

                # Sky-plane offsets (positive = East / North)
                ew_sign  = -1.0 if self.flip_ew else +1.0
                ns_sign  = +1.0 if self.flip_ns else -1.0
                east_px  = ew_sign * dra_arcsec  / plate_scale_arcsec_per_px
                north_px = ns_sign * ddec_arcsec / plate_scale_arcsec_per_px

                # Rotate into camera frame using effective camera PA = pole_pa + NP.ang
                # (auto_detect_pole_pa returns θ_cam − NP.ang, so +NP.ang recovers θ_cam)
                pa_rad = np.radians(pole_pa_deg + np_ang_deg)
                dx_px  = east_px * np.cos(pa_rad) + north_px * np.sin(pa_rad)
                dy_px  = east_px * np.sin(pa_rad) - north_px * np.cos(pa_rad)

                x_px   = disk_cx + dx_px
                y_px   = disk_cy + dy_px
                dist   = float(np.hypot(dx_px, dy_px))
                on_disk = dist < disk_radius_px

                positions.append(SatellitePos(
                    name=moon_name, x_px=float(x_px), y_px=float(y_px),
                    on_disk=on_disk, dist_px=dist,
                ))

            n_transit = sum(1 for p in positions if p.on_disk)
            if n_transit > 0:
                print(
                    f"  [SatTracker] {moon_name}: {n_transit}/{len(t_list)} frames"
                    f" on disk → TRANSIT DETECTED"
                )
            else:
                print(f"  [SatTracker] {moon_name}: off disk ({positions[0].dist_px:.0f}px"
                      f"–{positions[-1].dist_px:.0f}px from center)")

            results[moon_name] = positions

        return results

    def any_on_disk(self, positions: Dict[str, List[SatellitePos]]) -> bool:
        """Return True if any moon is on disk in any frame."""
        return any(
            p.on_disk
            for moon_pos in positions.values()
            for p in moon_pos
        )

    # ── Horizons queries ───────────────────────────────────────────────────────

    def _query_ra_dec(
        self,
        command: str,
        t_start: datetime,
        t_end: datetime,
        step_minutes: int = 2,
        observer_code: Optional[str] = None,
    ) -> List[Tuple[datetime, float, float]]:
        """Query RA/Dec (decimal degrees) from Horizons, with in-memory caching."""
        obs = observer_code or self.observer_code
        cache_key = f"{command}:{obs}:{t_start.strftime('%Y%m%d%H%M')}:{t_end.strftime('%Y%m%d%H%M')}"
        if cache_key in self._ra_dec_cache:
            return self._ra_dec_cache[cache_key]

        start_str = t_start.strftime("%Y-%m-%d %H:%M")
        stop_str  = t_end.strftime("%Y-%m-%d %H:%M")
        params = urllib.parse.urlencode({
            "format":     "text",
            "COMMAND":    f"'{command}'",
            "OBJ_DATA":   "NO",
            "MAKE_EPHEM": "YES",
            "EPHEM_TYPE": "OBSERVER",
            "CENTER":     f"'{obs}'",
            "START_TIME": f"'{start_str}'",
            "STOP_TIME":  f"'{stop_str}'",
            "STEP_SIZE":  f"{step_minutes}m",
            "QUANTITIES": "1",
            "ANG_FORMAT": "DEG",
            "EXTRA_PREC": "YES",
        })
        url = f"https://ssd.jpl.nasa.gov/api/horizons.api?{params}"
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                text = resp.read().decode("utf-8")
        except Exception as exc:
            warnings.warn(f"[SatTracker] Horizons RA/Dec query failed for {command}: {exc}")
            return []

        result = _parse_horizons_ra_dec(text)
        if result:
            self._ra_dec_cache[cache_key] = result
        return result

    def get_shadow_positions(
        self,
        t_list: List[datetime],
        disk_cx: float,
        disk_cy: float,
        disk_radius_px: float,
        plate_scale_arcsec_per_px: Optional[float] = None,
        pole_pa_deg: float = 0.0,
        np_ang_deg: float = 0.0,
        moon_horizons_positions: Optional[Dict[str, List["SatellitePos"]]] = None,
        time_offset_sec: float = 0.0,
    ) -> Dict[str, List["SatellitePos"]]:
        """Return pixel positions of Galilean moon SHADOWS via Skyfield oblate-spheroid intersection.

        Shoots a ray from the Sun through each moon and intersects Jupiter's oblate
        spheroid (R_eq=71492 km, R_pol=66854 km).  Transit detection is purely
        geometric — no Horizons /t flag needed.

        If moon_horizons_positions is provided ({moon_name: [SatellitePos, ...]}
        from get_positions()), the function computes the systematic Skyfield→Horizons
        calibration offset for each moon (mean Horizons−Skyfield over on-disk frames)
        and applies it to the corresponding shadow positions.

        Returns {"{moon}_shadow": [SatellitePos, ...]} for moons with detected shadow
        transits.  Requires Skyfield + de440s.bsp + jup365.bsp in /tmp/skyfield/.
        """
        if not t_list:
            return {}

        t_list_naive = [t.replace(tzinfo=None) if t.tzinfo is not None else t
                        for t in t_list]

        if plate_scale_arcsec_per_px is None:
            plate_scale_arcsec_per_px = self.get_plate_scale(disk_radius_px, t_list_naive[0])

        sf = _load_skyfield_kernels()
        if sf is None:
            warnings.warn("[SatTracker] Skyfield kernels unavailable — shadow positions skipped")
            return {}
        ts, eph, jup_moons = sf

        results: Dict[str, List[SatellitePos]] = {}

        for moon_name, sf_id in _MOON_SF_ID.items():
            try:
                moon_body = jup_moons[sf_id]
            except Exception:
                warnings.warn(f"[SatTracker] {moon_name} not found in {_SKYFIELD_MOONS_BSP}")
                continue

            shadow_key = f"{moon_name}_shadow"
            shadow_positions: List[SatellitePos] = []
            moon_sf_positions: List[SatellitePos] = []

            for t in t_list_naive:
                shad_pos, moon_sf_pos = _shadow_pos_skyfield(
                    t, moon_body, eph, jup_moons, ts,
                    disk_cx, disk_cy, disk_radius_px,
                    plate_scale_arcsec_per_px,
                    self.flip_ew, self.flip_ns,
                    shadow_key,
                    pole_pa_deg=pole_pa_deg,
                    np_ang_deg=np_ang_deg,
                    time_offset_sec=time_offset_sec,
                )
                shadow_positions.append(shad_pos)
                moon_sf_positions.append(moon_sf_pos)

            # ── Horizons calibration: apply (Horizons − Skyfield) offset ────
            cal_dx, cal_dy = 0.0, 0.0
            if moon_horizons_positions and moon_name in moon_horizons_positions:
                h_list = moon_horizons_positions[moon_name]
                deltas = [
                    (h.x_px - s.x_px, h.y_px - s.y_px)
                    for h, s in zip(h_list, moon_sf_positions)
                    if h.on_disk and s.on_disk
                ]
                if deltas:
                    cal_dx = float(np.mean([d[0] for d in deltas]))
                    cal_dy = float(np.mean([d[1] for d in deltas]))
                    print(
                        f"  [SatTracker] {moon_name} Horizons calibration:"
                        f" Δx={cal_dx:+.2f}px  Δy={cal_dy:+.2f}px"
                        f"  (from {len(deltas)} on-disk frames)"
                    )
                    shadow_positions = [
                        SatellitePos(
                            name=p.name,
                            x_px=p.x_px + cal_dx,
                            y_px=p.y_px + cal_dy,
                            on_disk=p.on_disk,
                            dist_px=p.dist_px,
                        )
                        for p in shadow_positions
                    ]

            n_transit = sum(1 for p in shadow_positions if p.on_disk)
            if n_transit > 0:
                print(
                    f"  [SatTracker] {moon_name} SHADOW: {n_transit}/{len(t_list)} frames"
                    f" on disk → SHADOW TRANSIT (Skyfield 3D oblate)"
                )
                results[shadow_key] = shadow_positions
            else:
                print(f"  [SatTracker] {moon_name} shadow: off disk (Skyfield 3D oblate)")

        return results

    def _query_angular_radius_arcsec(self, t_sample: datetime) -> Optional[float]:
        """Query Jupiter's angular diameter from Horizons and return angular radius."""
        start_str = t_sample.strftime("%Y-%m-%d %H:%M")
        stop_str  = (t_sample + timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M")
        params = urllib.parse.urlencode({
            "format":     "text",
            "COMMAND":    f"'{self.jupiter_id}'",
            "OBJ_DATA":   "NO",
            "MAKE_EPHEM": "YES",
            "EPHEM_TYPE": "OBSERVER",
            "CENTER":     f"'{self.observer_code}'",
            "START_TIME": f"'{start_str}'",
            "STOP_TIME":  f"'{stop_str}'",
            "STEP_SIZE":  "1m",
            "QUANTITIES": "13",   # angular diameter
        })
        url = f"https://ssd.jpl.nasa.gov/api/horizons.api?{params}"
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                text = resp.read().decode("utf-8")
        except Exception as exc:
            warnings.warn(f"[SatTracker] Horizons ang-diam query failed: {exc}")
            return None

        # Search for a floating-point number after $$SOE
        soe = text.find("$$SOE")
        eoe = text.find("$$EOE")
        if soe < 0 or eoe < 0:
            return None
        data_section = text[soe + 5: eoe]
        # ang-diam is the last numerical field on the data line
        for line in data_section.split("\n"):
            line = line.strip()
            if not line or line.startswith("*"):
                continue
            numbers = re.findall(r"-?\d+\.?\d*", line[20:])   # skip timestamp
            if numbers:
                try:
                    ang_diam = float(numbers[-1])
                    if 10.0 < ang_diam < 60.0:   # sanity: Jupiter is 30-50" typically
                        return ang_diam / 2.0
                except ValueError:
                    pass
        return None


# ── CV-based fallback ──────────────────────────────────────────────────────────

def detect_satellites_cv(
    image: np.ndarray,
    disk_cx: float,
    disk_cy: float,
    disk_radius_px: float,
    bright_threshold_frac: float = 0.97,
    dark_threshold_frac: float   = 0.10,
    min_radius_px: float = 3.0,
    max_radius_px: float = 30.0,
    local_bg_ring_inner: float   = 1.5,   # inner ring radius as multiple of blob radius
    local_bg_ring_outer: float   = 3.0,   # outer ring radius as multiple of blob radius
    min_local_contrast: float    = 0.08,  # blob peak must exceed local bg by this fraction
    min_circularity: float       = 0.55,  # 4π·area/perimeter² — avoid elongated features
) -> List[SatellitePos]:
    """Detect satellite/shadow blobs inside the disk using strict thresholding.

    Satellites are small, nearly circular, and significantly brighter than their
    local neighbourhood (not just the disk-wide peak).  Jupiter's atmospheric
    features (EZ, STB, etc.) are large and diffuse — rejected by size + local
    contrast checks.

    Shadows appear as small dark circular blobs.

    Returns a list of SatellitePos objects (name='cv_bright' or 'cv_shadow').
    This is used as a fallback when Horizons is unavailable.
    """
    lum = image.mean(axis=2).astype(np.float32) if image.ndim == 3 else image.astype(np.float32)
    # Normalise to [0, 1] so thresholds are scale-independent
    lum_min, lum_max = float(lum.min()), float(lum.max())
    if lum_max - lum_min < 1e-6:
        return []
    lum = (lum - lum_min) / (lum_max - lum_min)

    h, w = lum.shape

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist_from_center = np.sqrt((xx - disk_cx) ** 2 + (yy - disk_cy) ** 2)
    disk_mask = dist_from_center < disk_radius_px * 0.92   # exclude limb brightening

    disk_pixels = lum[disk_mask]
    if disk_pixels.size == 0:
        return []

    peak   = float(disk_pixels.max())
    p05    = float(np.percentile(disk_pixels, 5))   # near-dark reference

    detections: List[SatellitePos] = []

    for label, threshold, above in [
        ("cv_bright", peak   * bright_threshold_frac, True),
        ("cv_shadow", p05    + (peak - p05) * dark_threshold_frac, False),
    ]:
        if above:
            binary = ((lum >= threshold) & disk_mask).astype(np.uint8) * 255
        else:
            binary = ((lum <= threshold) & disk_mask).astype(np.uint8) * 255

        # Morphological open to remove isolated noise pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < np.pi * min_radius_px ** 2:
                continue
            r = float(np.sqrt(area / np.pi))
            if r > max_radius_px:
                continue

            # Circularity check — reject elongated atmospheric streaks
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1.0:
                continue
            circularity = 4.0 * np.pi * area / (perimeter ** 2)
            if circularity < min_circularity:
                continue

            M = cv2.moments(cnt)
            if M["m00"] < 1e-6:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            dist = float(np.hypot(cx - disk_cx, cy - disk_cy))
            if dist >= disk_radius_px * 0.95:
                continue

            # Local contrast check: blob brightness vs annular neighbourhood
            inner_r = r * local_bg_ring_inner
            outer_r = r * local_bg_ring_outer
            dist_map = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            ring_mask = (dist_map >= inner_r) & (dist_map < outer_r) & disk_mask
            blob_mask = dist_map < r

            ring_pixels = lum[ring_mask]
            blob_pixels = lum[blob_mask]
            if ring_pixels.size < 4 or blob_pixels.size < 1:
                continue

            local_bg   = float(np.median(ring_pixels))
            blob_value = float(np.median(blob_pixels))

            if above:
                if (blob_value - local_bg) < min_local_contrast:
                    continue   # not significantly brighter than surroundings
            else:
                if (local_bg - blob_value) < min_local_contrast:
                    continue   # not significantly darker than surroundings

            detections.append(SatellitePos(
                name=label, x_px=float(cx), y_px=float(cy),
                on_disk=True, dist_px=dist,
            ))

    return detections


# ── Helper: make a feathered circular satellite mask ──────────────────────────

def make_satellite_mask(
    shape: Tuple[int, int],
    positions: List[SatellitePos],
    mask_radius_px: float = 25.0,
    feather_px: float = 6.0,
) -> np.ndarray:
    """Create a [0, 1] float32 mask. 0.0 = satellite pixel, 1.0 = valid planet pixel.

    Multiple satellite positions are combined by taking the per-pixel minimum.
    """
    h, w = shape
    mask = np.ones((h, w), dtype=np.float32)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for pos in positions:
        if not pos.on_disk:
            continue
        dist = np.sqrt((xx - pos.x_px) ** 2 + (yy - pos.y_px) ** 2)
        # 0.0 inside (radius - feather), linearly ramping to 1.0 at radius
        sat_mask = np.clip((dist - (mask_radius_px - feather_px)) / feather_px, 0.0, 1.0)
        mask = np.minimum(mask, sat_mask)

    return mask


# ── Diagnostic overlay ────────────────────────────────────────────────────────

def save_diagnostic_overlay(
    image: np.ndarray,
    disk_cx: float,
    disk_cy: float,
    disk_radius_px: float,
    sat_positions: Dict[str, List[SatellitePos]],
    out_path: Path,
    frame_idx: int = 0,
    cv_detections: Optional[List[SatellitePos]] = None,
    pole_pa_deg: Optional[float] = None,
) -> None:
    """Save an annotated PNG showing satellite positions on the reference frame.

    Draws:
      - Disk circle (white) + cross at center
      - Jupiter rotation axis (cyan dashed line, PA from image-up toward image-right)
      - Trajectory polylines for each moon across all frames (body=green, shadow=orange)
      - Current-frame position circle + label
      - CV detections (yellow)
    """
    # Normalise to uint8
    if image.dtype == np.float32 or image.dtype == np.float64:
        vis = np.clip(image * 255, 0, 255).astype(np.uint8)
    elif image.dtype == np.uint16:
        vis = (image >> 8).astype(np.uint8)
    else:
        vis = image.astype(np.uint8)

    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    elif vis.ndim == 3 and vis.shape[2] == 3:
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    h, w = vis.shape[:2]

    # Disk outline + center cross
    cv2.circle(vis, (int(disk_cx), int(disk_cy)), int(disk_radius_px),
               color=(200, 200, 200), thickness=1, lineType=cv2.LINE_AA)
    cv2.drawMarker(vis, (int(disk_cx), int(disk_cy)), color=(200, 200, 200),
                   markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)

    # ── Jupiter rotation axis ────────────────────────────────────────────────────
    if pole_pa_deg is not None:
        pa_rad = np.radians(pole_pa_deg)
        # pole_pa_deg: angle of Jupiter's north pole from image-up toward image-right
        # North pole unit vector in image coords (y-axis down):
        #   dx = +sin(pa)  (rightward = East)
        #   dy = -cos(pa)  (upward = -y)
        ax_len = disk_radius_px * 1.35
        dx_n =  np.sin(pa_rad) * ax_len
        dy_n = -np.cos(pa_rad) * ax_len
        p_n = (int(disk_cx + dx_n), int(disk_cy + dy_n))   # North pole end
        p_s = (int(disk_cx - dx_n), int(disk_cy - dy_n))   # South pole end
        # Draw as dashed cyan line (simulate dashes with segments)
        axis_color = (220, 200, 0)   # cyan-ish (BGR)
        n_dashes = 12
        for i in range(n_dashes):
            t0 = i / n_dashes
            t1 = (i + 0.55) / n_dashes
            x0 = int(p_s[0] + t0 * (p_n[0] - p_s[0]))
            y0 = int(p_s[1] + t0 * (p_n[1] - p_s[1]))
            x1 = int(p_s[0] + t1 * (p_n[0] - p_s[0]))
            y1 = int(p_s[1] + t1 * (p_n[1] - p_s[1]))
            cv2.line(vis, (x0, y0), (x1, y1), axis_color, thickness=1, lineType=cv2.LINE_AA)
        # "N" label near north pole end (offset slightly outward)
        label_x = int(disk_cx + dx_n * 1.05) - 5
        label_y = int(disk_cy + dy_n * 1.05) + 5
        cv2.putText(vis, "N", (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, axis_color, 1, cv2.LINE_AA)

    # ── Trajectory polylines (all frames) ───────────────────────────────────────
    for moon_name, pos_list in sat_positions.items():
        is_shadow = moon_name.endswith("_shadow")
        traj_color = (0, 100, 200) if is_shadow else (0, 180, 0)   # dim orange/green for trail

        valid_pts = []
        for p in pos_list:
            if p is None:
                continue
            valid_pts.append((int(round(p.x_px)), int(round(p.y_px))))

        if len(valid_pts) >= 2:
            for i in range(len(valid_pts) - 1):
                cv2.line(vis, valid_pts[i], valid_pts[i + 1],
                         traj_color, thickness=1, lineType=cv2.LINE_AA)
            # Small dots at each waypoint
            for pt in valid_pts:
                cv2.circle(vis, pt, 3, traj_color, thickness=-1, lineType=cv2.LINE_AA)

    # ── Current-frame positions ──────────────────────────────────────────────────
    for moon_name, pos_list in sat_positions.items():
        if frame_idx >= len(pos_list) or pos_list[frame_idx] is None:
            continue
        pos = pos_list[frame_idx]
        ix, iy = int(round(pos.x_px)), int(round(pos.y_px))
        is_shadow = moon_name.endswith("_shadow")
        if is_shadow:
            color = (0, 140, 255) if pos.on_disk else (0, 80, 160)   # orange (BGR)
        else:
            color = (0, 220, 0)   if pos.on_disk else (0, 140, 0)    # green
        radius = 15 if is_shadow else 20
        cv2.circle(vis, (ix, iy), radius, color, thickness=1, lineType=cv2.LINE_AA)
        status = "SHADOW" if (is_shadow and pos.on_disk) else ("TRANSIT" if pos.on_disk else f"{pos.dist_px:.0f}px")
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Labels centered below the circle
        (nw, _), _ = cv2.getTextSize(moon_name, font, 0.45, 1)
        cv2.putText(vis, moon_name, (ix - nw // 2, iy + radius + 14),
                    font, 0.45, color, 1, cv2.LINE_AA)
        (sw, _), _ = cv2.getTextSize(status, font, 0.35, 1)
        cv2.putText(vis, status, (ix - sw // 2, iy + radius + 27),
                    font, 0.35, color, 1, cv2.LINE_AA)

    # ── CV detections (yellow) ───────────────────────────────────────────────────
    if cv_detections:
        for det in cv_detections:
            ix, iy = int(round(det.x_px)), int(round(det.y_px))
            label_color = (0, 200, 255) if det.name == "cv_bright" else (200, 100, 255)
            cv2.circle(vis, (ix, iy), 18, label_color, thickness=1, lineType=cv2.LINE_AA)
            cv2.putText(vis, det.name, (ix + 20, iy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, label_color, 1, cv2.LINE_AA)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    print(f"  [SatTracker] Diagnostic overlay → {out_path}")


# ── Horizons text parser ───────────────────────────────────────────────────────

def _parse_horizons_ra_dec(
    text: str,
) -> List[Tuple[datetime, float, float]]:
    """Parse RA (deg) and Dec (deg) from Horizons text with ANG_FORMAT=DEG."""
    soe = text.find("$$SOE")
    eoe = text.find("$$EOE")
    if soe < 0 or eoe < 0:
        return []

    data_section = text[soe + 5: eoe]
    results: List[Tuple[datetime, float, float]] = []

    # Pattern: timestamp field (fixed 18 chars) then RA_deg  Dec_deg  ...
    # "2026-May-05 10:00     157.12345  +12.34567 ..."
    # or with EXTRA_PREC: "2026-May-05 10:00     157.123456  +12.345678 ..."
    _MONTHS = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    }
    line_re  = re.compile(
        r"(\d{4})-([A-Za-z]{3})-(\d{2})\s+(\d{2}):(\d{2})"
        r"\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)"
    )
    for line in data_section.split("\n"):
        m = line_re.search(line)
        if not m:
            continue
        year, mon_str, day = int(m.group(1)), m.group(2).capitalize(), int(m.group(3))
        hour, minute       = int(m.group(4)), int(m.group(5))
        ra_deg             = float(m.group(6))
        dec_deg            = float(m.group(7))
        mon = _MONTHS.get(mon_str)
        if mon is None:
            continue
        try:
            t = datetime(year, mon, day, hour, minute)
        except ValueError:
            continue
        results.append((t, ra_deg, dec_deg))

    return results


# ── Skyfield shadow computation ────────────────────────────────────────────────

def _resolve_skyfield_dir() -> Path:
    """Return first directory that contains both BSP kernels, or primary dir (for download)."""
    candidates = [_SKYFIELD_KERNEL_DIR, Path("/tmp/skyfield")]
    for d in candidates:
        if (d / _SKYFIELD_PLANETS_BSP).exists() and (d / _SKYFIELD_MOONS_BSP).exists():
            return d
    return _SKYFIELD_KERNEL_DIR


def _download_bsp(name: str, dest_dir: Path) -> bool:
    """Download a BSP kernel from JPL NAIF. Prints progress. Returns True on success."""
    import urllib.request
    url = _BSP_URLS.get(name)
    if not url:
        return False
    dest = dest_dir / name
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"[SatTracker] Downloading {name} ({url})", flush=True)
    try:
        def _hook(count, block, total):
            mb = count * block / 1_048_576
            tot_mb = total / 1_048_576 if total > 0 else 0
            pct = int(count * block * 100 / total) if total > 0 else 0
            print(f"\r  {mb:.0f} / {tot_mb:.0f} MB  ({pct}%)", end="", flush=True)
        urllib.request.urlretrieve(url, str(dest), reporthook=_hook)
        print(f"\n[SatTracker] Saved → {dest}", flush=True)
        return True
    except Exception as exc:
        print(f"\n[SatTracker] Download failed: {exc}", flush=True)
        if dest.exists():
            dest.unlink()
        return False


def _load_skyfield_kernels():
    """Load Skyfield TimeScale, de440s.bsp, and jup365.bsp.  Returns (ts, eph, jup_moons) or None.

    Downloads missing BSP files from JPL NAIF on first use (may take several minutes for jup365.bsp).
    The GUI BspStatusRow widget warns the user before this happens.
    """
    try:
        from skyfield.api import Loader
    except ImportError:
        warnings.warn("[SatTracker] skyfield not installed — shadow positions unavailable")
        return None
    kernel_dir = _resolve_skyfield_dir()
    # Download any missing BSP files
    for bsp in [_SKYFIELD_PLANETS_BSP, _SKYFIELD_MOONS_BSP]:
        if not (kernel_dir / bsp).exists():
            if not _download_bsp(bsp, kernel_dir):
                warnings.warn(f"[SatTracker] Could not download {bsp} — shadow positions unavailable")
                return None
    try:
        load      = Loader(str(kernel_dir))
        ts        = load.timescale()
        eph       = load(_SKYFIELD_PLANETS_BSP)
        jup_moons = load(_SKYFIELD_MOONS_BSP)
        return ts, eph, jup_moons
    except Exception as exc:
        warnings.warn(f"[SatTracker] Skyfield kernel load error: {exc}")
        return None


def _shadow_pos_skyfield(
    t: datetime,
    moon_body,
    eph,
    jup_moons,
    ts,
    disk_cx: float,
    disk_cy: float,
    disk_radius_px: float,
    plate_scale: float,
    flip_ew: bool,
    flip_ns: bool,
    shadow_name: str,
    pole_pa_deg: float = 0.0,
    np_ang_deg: float = 0.0,
    time_offset_sec: float = 0.0,
) -> Tuple["SatellitePos", "SatellitePos"]:
    """3D ray-oblate-spheroid shadow intersection using Skyfield ephemeris.

    Applies light-travel-time (LTT) correction: shadow geometry is computed at
    t_emit = t_observe − d_EJ/c (≈ 47 min for Jupiter), matching the Horizons
    convention for apparent positions.  Earth's position is evaluated at t_observe
    for the final RA/Dec projection.

    Jupiter is modelled as an oblate spheroid (R_eq=71492 km, R_pol=66854 km)
    with its pole aligned to _JUP_POLE_ICRF.

    time_offset_sec: clock correction in seconds.  Applied to t before Skyfield
    query.  Use a negative value when the capture PC clock was fast.

    Returns (shadow_pos, moon_skyfield_pos).  shadow_pos.on_disk=True only when the
    near-hemisphere intersection is within disk_radius_px of the disk center.
    """
    t_corrected = t + timedelta(seconds=time_offset_sec)
    t_sf = ts.utc(t_corrected.year, t_corrected.month, t_corrected.day,
                  t_corrected.hour, t_corrected.minute, t_corrected.second)

    # Earth position at observation time (for final RA/Dec projection)
    earth_km = eph['earth'].at(t_sf).position.km               # (3,)

    # Light travel time from Jupiter to Earth — emit time for shadow geometry
    jup_km_t = eph['jupiter barycenter'].at(t_sf).position.km
    d_EJ_km  = float(np.linalg.norm(jup_km_t - earth_km))
    lt_days  = d_EJ_km / (299792.458 * 86400.0)   # km / (km/day)
    t_emit   = ts.tt_jd(float(t_sf.tt) - lt_days)

    # ICRF positions at emission time (shadow geometry — LTT corrected)
    # jup_km must use the same kernel as moon_km so ray geometry is self-consistent.
    sun_km  = eph['sun'].at(t_emit).position.km                # (3,)
    jup_km  = jup_moons['jupiter barycenter'].at(t_emit).position.km  # jup365, same frame as moon_km
    moon_km = moon_body.at(t_emit).position.km                 # ICRF from SSB (auto-chained)

    moon_name_body = shadow_name.replace("_shadow", "")
    _off_shad = SatellitePos(name=shadow_name, x_px=disk_cx, y_px=disk_cy,
                             on_disk=False, dist_px=disk_radius_px * 10.0)
    _off_moon = SatellitePos(name=moon_name_body, x_px=disk_cx, y_px=disk_cy,
                             on_disk=False, dist_px=disk_radius_px * 10.0)

    # ── Oblate-spheroid ray intersection ──────────────────────────────────────
    # Ray: P(λ) = sun_km + λ*(moon_km − sun_km),  λ=0 at Sun, λ=1 at moon
    # Jupiter pole unit vector (ICRF, approximately constant)
    pole = _JUP_POLE_ICRF   # (3,) float64

    ray_d = moon_km - sun_km   # direction (km)
    w     = sun_km  - jup_km   # Sun offset from Jupiter center

    # Decompose ray_d and w into polar (z) and equatorial components
    D_z   = float(np.dot(ray_d, pole))
    D_perp = ray_d - D_z * pole   # equatorial component of ray direction
    W_z   = float(np.dot(w, pole))
    W_perp = w - W_z * pole        # equatorial component of Sun-Jupiter offset

    a_sq = float(_JUP_EQ_RADIUS_KM ** 2)
    c_sq = float(_JUP_POL_RADIUS_KM ** 2)

    A_obl = float(np.dot(D_perp, D_perp)) / a_sq + D_z ** 2 / c_sq
    B_obl = 2.0 * (float(np.dot(W_perp, D_perp)) / a_sq + W_z * D_z / c_sq)
    C_obl = float(np.dot(W_perp, W_perp)) / a_sq + W_z ** 2 / c_sq - 1.0

    disc = B_obl * B_obl - 4.0 * A_obl * C_obl
    if disc < 0.0:
        return _off_shad, _off_moon   # shadow ray misses Jupiter spheroid

    # Disk-plane intersection: where the shadow ray crosses the plane perpendicular
    # to the Earth-Jupiter line-of-sight passing through Jupiter's center.
    # This gives the apparent shadow position on Jupiter's projected disk,
    # accounting for viewing geometry. Reduces positional error from ~17px to <1px
    # compared to lam_near (oblate spheroid surface intersection).
    e_hat = earth_km - jup_km
    e_hat = e_hat / float(np.linalg.norm(e_hat))
    W_e = float(np.dot(w, e_hat))
    D_e = float(np.dot(ray_d, e_hat))
    if abs(D_e) < 1e-10:
        return _off_shad, _off_moon
    lam_plane = -W_e / D_e

    # λ=1 is the moon's position.  Valid shadow: λ > 1 (Sun → Moon → Jupiter).
    if lam_plane <= 1.0:
        return _off_shad, _off_moon

    shadow_km = sun_km + lam_plane * ray_d

    # ── Shared coordinate conversion helper ───────────────────────────────────
    def _ra_dec(pos_km: np.ndarray) -> Tuple[float, float]:
        d = pos_km - earth_km
        d = d / float(np.linalg.norm(d))
        dec = float(np.degrees(np.arcsin(float(np.clip(d[2], -1.0, 1.0)))))
        ra  = float(np.degrees(np.arctan2(float(d[1]), float(d[0]))) % 360.0)
        return ra, dec

    def _icrf_to_px(ra: float, dec: float, ref_ra: float, ref_dec: float) -> Tuple[float, float, float, float]:
        dra_deg = ra - ref_ra
        if dra_deg >  180.0: dra_deg -= 360.0
        if dra_deg < -180.0: dra_deg += 360.0
        dra_arcsec  = dra_deg * np.cos(np.radians(ref_dec)) * 3600.0
        ddec_arcsec = (dec - ref_dec) * 3600.0
        ew_sign  = -1.0 if flip_ew else +1.0
        ns_sign  = +1.0 if flip_ns else -1.0
        east_px  = ew_sign * dra_arcsec  / plate_scale
        north_px = ns_sign * ddec_arcsec / plate_scale
        pa_rad   = np.radians(pole_pa_deg + np_ang_deg)
        dx_px    = east_px * np.cos(pa_rad) + north_px * np.sin(pa_rad)
        dy_px    = east_px * np.sin(pa_rad) - north_px * np.cos(pa_rad)
        return disk_cx + dx_px, disk_cy + dy_px, dx_px, dy_px

    jup_ra, jup_dec = _ra_dec(jup_km)

    sha_ra, sha_dec = _ra_dec(shadow_km)
    shad_x, shad_y, shad_dx, shad_dy = _icrf_to_px(sha_ra, sha_dec, jup_ra, jup_dec)
    shad_dist = float(np.hypot(shad_dx, shad_dy))

    moon_ra, moon_dec = _ra_dec(moon_km)
    moon_x, moon_y, moon_dx, moon_dy = _icrf_to_px(moon_ra, moon_dec, jup_ra, jup_dec)
    moon_dist = float(np.hypot(moon_dx, moon_dy))

    shadow_pos = SatellitePos(name=shadow_name,
                              x_px=float(shad_x), y_px=float(shad_y),
                              on_disk=shad_dist < disk_radius_px, dist_px=shad_dist)
    moon_pos   = SatellitePos(name=moon_name_body,
                              x_px=float(moon_x), y_px=float(moon_y),
                              on_disk=moon_dist < disk_radius_px, dist_px=moon_dist)
    return shadow_pos, moon_pos


def _interp_ra_dec(
    ephem: List[Tuple[datetime, float, float]],
    t: datetime,
) -> Tuple[float, float]:
    """Linearly interpolate RA/Dec at time t from a sorted ephemeris list."""
    if not ephem:
        return 0.0, 0.0
    if t <= ephem[0][0]:
        return ephem[0][1], ephem[0][2]
    if t >= ephem[-1][0]:
        return ephem[-1][1], ephem[-1][2]
    # Binary search for the bracketing interval
    lo, hi = 0, len(ephem) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ephem[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    t0, ra0, dec0 = ephem[lo]
    t1, ra1, dec1 = ephem[hi]
    dt_total = (t1 - t0).total_seconds()
    if dt_total < 1e-3:
        return ra0, dec0
    frac = (t - t0).total_seconds() / dt_total
    # RA needs circular interpolation near 0/360
    dra = (ra1 - ra0 + 540.0) % 360.0 - 180.0
    return (ra0 + frac * dra) % 360.0, dec0 + frac * (dec1 - dec0)


# ── CV position refinement ────────────────────────────────────────────────────

def _local_extremum_centroid(
    lum: np.ndarray,
    cx: float, cy: float,
    search_radius: float,
    is_shadow: bool,
    local_percentile: float = 20.0,
    min_blob_area_px: float = 12.0,
) -> Optional[Tuple[float, float]]:
    """Find the centroid of the brightest (body) or darkest (shadow) cluster
    within a local patch around (cx, cy).  Uses local percentile thresholding
    so it works even when the satellite is dim relative to the global disk.

    Returns (refined_x, refined_y) in full-image pixel coordinates, or None.
    """
    import cv2 as _cv2
    h, w = lum.shape[:2]
    x1 = max(0, int(cx - search_radius))
    x2 = min(w, int(cx + search_radius + 1))
    y1 = max(0, int(cy - search_radius))
    y2 = min(h, int(cy + search_radius + 1))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None

    crop = lum[y1:y2, x1:x2].astype(np.float32)

    if is_shadow:
        thr = float(np.percentile(crop, local_percentile))
        binary = ((crop <= thr) * 255).astype(np.uint8)
    else:
        thr = float(np.percentile(crop, 100.0 - local_percentile))
        binary = ((crop >= thr) * 255).astype(np.uint8)

    kernel = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (3, 3))
    binary = _cv2.morphologyEx(binary, _cv2.MORPH_OPEN, kernel)

    contours, _ = _cv2.findContours(binary, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    patch_cx = (x2 - x1) / 2.0
    patch_cy = (y2 - y1) / 2.0
    best_pt, best_d = None, float("inf")
    for cnt in contours:
        area = _cv2.contourArea(cnt)
        if area < min_blob_area_px:
            continue
        M = _cv2.moments(cnt)
        if M["m00"] < 1e-6:
            continue
        bx = M["m10"] / M["m00"]
        by = M["m01"] / M["m00"]
        d  = float(np.hypot(bx - patch_cx, by - patch_cy))
        if d < best_d:
            best_d, best_pt = d, (bx + x1, by + y1)

    return best_pt


def refine_positions_with_cv(
    ref_lum: np.ndarray,
    disk_cx: float,
    disk_cy: float,
    disk_radius_px: float,
    sat_positions: Dict[str, List[SatellitePos]],
    search_radius_px: float = 35.0,
) -> Tuple[Dict[str, List[SatellitePos]], Dict[str, Tuple[float, float]]]:
    """Refine Horizons-predicted positions using local-patch extremum search.

    For each on-disk satellite/shadow, finds the brightest (body) or darkest
    (shadow) cluster within search_radius_px of the Horizons prediction.
    The systematic offset is applied uniformly to all frames for that satellite.

    Returns:
        refined_positions: same structure as sat_positions, with corrected centers.
        deltas_xy: {name: (dx, dy)} — (0.0, 0.0) if no match found.
    """
    refined:   Dict[str, List[SatellitePos]]  = {}
    deltas_xy: Dict[str, Tuple[float, float]] = {}

    for name, positions in sat_positions.items():
        is_shadow = name.endswith("_shadow")

        ref_idx = next((i for i, p in enumerate(positions) if p is not None and p.on_disk), None)
        if ref_idx is None:
            refined[name]   = positions
            deltas_xy[name] = (0.0, 0.0)
            continue

        ref_pos = positions[ref_idx]
        result  = _local_extremum_centroid(
            ref_lum, ref_pos.x_px, ref_pos.y_px,
            search_radius=search_radius_px,
            is_shadow=is_shadow,
        )

        if result is None:
            refined[name]   = positions
            deltas_xy[name] = (0.0, 0.0)
            print(f"  [CV refine] {name}: local search failed → Horizons kept")
            continue

        rx, ry = result
        dx = rx - ref_pos.x_px
        dy = ry - ref_pos.y_px
        dist = float(np.hypot(dx, dy))

        if dist > search_radius_px * 0.9:
            refined[name]   = positions
            deltas_xy[name] = (0.0, 0.0)
            print(f"  [CV refine] {name}: correction {dist:.1f}px too large → Horizons kept")
            continue

        print(f"  [CV refine] {name}: Δ=({dx:+.1f},{dy:+.1f})px  dist={dist:.1f}px → corrected")

        corrected = []
        for pos in positions:
            if pos is None or not pos.on_disk:
                corrected.append(pos)
            else:
                corrected.append(SatellitePos(
                    name=pos.name,
                    x_px=pos.x_px + dx, y_px=pos.y_px + dy,
                    on_disk=True, dist_px=pos.dist_px,
                ))
        refined[name]   = corrected
        deltas_xy[name] = (dx, dy)

    return refined, deltas_xy


def apply_cv_offsets(
    sat_positions: Dict[str, List[SatellitePos]],
    offsets: Dict[str, Tuple[float, float]],
) -> Dict[str, List[SatellitePos]]:
    """Apply pre-computed (dx, dy) offsets to Horizons positions."""
    refined: Dict[str, List[SatellitePos]] = {}
    for name, positions in sat_positions.items():
        dx, dy = offsets.get(name, (0.0, 0.0))
        if dx == 0.0 and dy == 0.0:
            refined[name] = positions
            continue
        dist = float(np.hypot(dx, dy))
        print(f"  [CV refine] {name}: Δ=({dx:+.1f},{dy:+.1f})px  dist={dist:.1f}px → applied (pre-computed)")
        corrected = []
        for pos in positions:
            if pos is None or not pos.on_disk:
                corrected.append(pos)
            else:
                corrected.append(SatellitePos(
                    name=pos.name,
                    x_px=pos.x_px + dx, y_px=pos.y_px + dy,
                    on_disk=True, dist_px=pos.dist_px,
                ))
        refined[name] = corrected
    return refined


def average_body_shadow_offsets(
    offsets: Dict[str, Tuple[float, float]],
) -> Dict[str, Tuple[float, float]]:
    """Average body + shadow corrections for the same moon when both are valid.

    When both the body (e.g. 'Europa') and its shadow ('Europa_shadow') have
    non-zero CV corrections, replaces both with their mean, giving a more
    robust estimate than either measurement alone.
    """
    averaged = dict(offsets)
    for body_key in list(offsets):
        if body_key.endswith("_shadow"):
            continue
        shadow_key = f"{body_key}_shadow"
        if shadow_key not in offsets:
            continue
        bdx, bdy = offsets[body_key]
        sdx, sdy = offsets[shadow_key]
        if (bdx != 0.0 or bdy != 0.0) and (sdx != 0.0 or sdy != 0.0):
            adx, ady = (bdx + sdx) / 2.0, (bdy + sdy) / 2.0
            print(
                f"  [CV avg/{body_key}] body=({bdx:+.1f},{bdy:+.1f})"
                f" shadow=({sdx:+.1f},{sdy:+.1f})"
                f" → avg=({adx:+.1f},{ady:+.1f})"
            )
            averaged[body_key]   = (adx, ady)
            averaged[shadow_key] = (adx, ady)
    return averaged
