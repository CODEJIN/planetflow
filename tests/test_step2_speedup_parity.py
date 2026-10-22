"""Bit-exact parity tests for the 2026-10-22 Step 2 (lucky stacking) speed-ups.

Every speed-up in that change is meant to leave the numbers untouched, so
each test pins the new code against a frozen copy of the code it replaced
(the `_ref_*` functions below, copied verbatim from before the change) or
against an independent path, with exact equality — no tolerance:

  * derotation._gradient_disk_r / _subpixel_ray_edge: per-ray Python loop
    → one (n_rays, n_samples) batch.
  * lucky_stack.score_frames_log_disk: sequential loop → thread pool.
  * lucky_stack._batch_qsf_refine: gather 3×3 neighbourhoods before the
    float64 cast instead of casting the whole batch.
  * lucky_stack._per_ap_independent_stack Pass 1: per-frame body moved to
    _pass1_frame; run in worker processes over shared memory; the aligned
    frames Pass 1 computes are reused instead of being recomputed.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import APGridMode, LuckyStackConfig
from pipeline.modules import derotation
from pipeline.modules import lucky_stack as ls
from pipeline.modules.derotation import (
    _bilinear_interp,
    _gaussian_filter1d_np,
    apply_shift,
    limb_center_align,
    subpixel_align,
)
from pipeline.modules.ser_io import SERReader, SERWriter


# ── Frozen pre-change implementations ─────────────────────────────────────────

def _ref_subpixel_ray_edge(profile, dr, smooth_sigma=1.5, margin=None):
    smoothed = _gaussian_filter1d_np(profile, sigma=smooth_sigma)
    grad = np.gradient(smoothed, dr)
    if margin is None:
        margin = max(3, len(grad) // 20)
    if margin <= 0:
        search = grad
    else:
        if len(grad) <= 2 * margin:
            return None
        search = grad[margin:-margin]
    idx = margin + int(np.argmin(search))
    if idx <= 0 or idx >= len(grad) - 1:
        return None
    y0, y1, y2 = grad[idx - 1], grad[idx], grad[idx + 1]
    denom = 2.0 * (y2 - 2.0 * y1 + y0)
    sub = -(y2 - y0) / denom if abs(denom) > 1e-12 else 0.0
    return idx + sub


def _ref_gradient_disk_r(image, cx, cy, r_rough, n_rays=72, search_frac=(0.75, 1.30),
                         n_samples=100, smooth_sigma=1.5, outlier_sigma=2.0,
                         return_n_valid=False):
    h, w = image.shape[:2]
    angles = np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False)
    r_start = r_rough * search_frac[0]
    r_end = r_rough * search_frac[1]
    r_vals = np.linspace(r_start, r_end, n_samples)
    dr = r_vals[1] - r_vals[0]
    edge_radii = []
    for angle in angles:
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        xs = cx + r_vals * cos_a
        ys = cy + r_vals * sin_a
        if (xs < 0).any() or (xs >= w - 1).any() or \
           (ys < 0).any() or (ys >= h - 1).any():
            continue
        profile = _bilinear_interp(image, ys, xs)
        sub_idx = _ref_subpixel_ray_edge(profile, dr, smooth_sigma=smooth_sigma, margin=0)
        if sub_idx is None:
            continue
        edge_radii.append(float(r_vals[0] + sub_idx * dr))
    if len(edge_radii) < 8:
        return (r_rough, len(edge_radii)) if return_n_valid else r_rough
    arr = np.array(edge_radii)
    med = float(np.median(arr))
    std = float(np.std(arr))
    keep = arr[np.abs(arr - med) < outlier_sigma * (std + 0.5)]
    if len(keep) < 6:
        keep = arr
    result = float(np.median(keep))
    return (result, len(edge_radii)) if return_n_valid else result


def _ref_score_frames_log_disk(reader, cfg, score_step=2):
    n_frames = reader.header["FrameCount"]
    sigma = float(getattr(cfg, "log_disk_sigma", 3.0))
    thr = float(getattr(cfg, "log_disk_threshold", 0.25))
    sampled_idx = list(range(0, n_frames, score_step))
    sampled_scores = []
    for idx in sampled_idx:
        raw = reader.get_frame(idx)
        f = raw.astype(np.float32) / raw.max()
        mask = f > thr
        if mask.sum() < 50:
            sampled_scores.append(0.0)
        else:
            blurred = cv2.GaussianBlur(f, (0, 0), sigma)
            lap = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
            sampled_scores.append(float(lap[mask].var()))
    return np.interp(
        np.arange(n_frames, dtype=np.float32),
        np.array(sampled_idx, dtype=np.float32),
        np.array(sampled_scores, dtype=np.float32),
    ).astype(np.float32)


def _ref_pass1(frames, reference, ap_positions, disk_cx, disk_cy, disk_radius, cfg):
    """Pre-change Pass 1 loop + the intermediate apply_shift re-run."""
    H, W = reference.shape[:2]
    n_sel, n_ap = len(frames), len(ap_positions)
    ksize = int(getattr(cfg, "quality_gradient_ksize", 3))
    stab = int(getattr(cfg, "stabilization_planet_threshold", 0))
    metric = str(getattr(cfg, "score_metric", "local_gradient"))
    log_sigma = float(getattr(cfg, "log_disk_sigma", 3.0))
    log_thr = float(getattr(cfg, "log_disk_threshold", 0.25))
    global_shifts = np.zeros((n_sel, 2), dtype=np.float32)
    score_matrix = np.zeros((n_sel, n_ap), dtype=np.float32)
    for i, frame in enumerate(frames):
        dx_g, dy_g = limb_center_align(disk_cx, disk_cy, frame, fixed_threshold=stab)
        if abs(dx_g) > disk_radius * 0.5 or abs(dy_g) > disk_radius * 0.5:
            dx_g, dy_g = subpixel_align(reference, frame)
        global_shifts[i] = (dx_g, dy_g)
        aligned = apply_shift(frame, dx_g, dy_g)
        if metric != "log_disk":
            gx = cv2.Sobel(aligned, cv2.CV_32F, 1, 0, ksize=ksize)
            gy = cv2.Sobel(aligned, cv2.CV_32F, 0, 1, ksize=ksize)
            mag2 = gx ** 2 + gy ** 2
        for j, ap in enumerate(ap_positions):
            ax, ay = int(ap[0]), int(ap[1])
            half = (int(ap[2]) if len(ap) >= 3 else cfg.ap_size) // 2
            if ay - half < 0 or ay + half > H or ax - half < 0 or ax + half > W:
                continue
            patch = aligned[ay - half: ay + half, ax - half: ax + half]
            if metric == "log_disk":
                pf = patch.astype(np.float32)
                pm = float(pf.max())
                if pm > 1e-9:
                    pf /= pm
                mask_p = pf > log_thr
                if mask_p.sum() < 5:
                    score_matrix[i, j] = 0.0
                else:
                    bl = cv2.GaussianBlur(pf, (0, 0), log_sigma)
                    lp = cv2.Laplacian(bl, cv2.CV_32F, ksize=3)
                    score_matrix[i, j] = float(lp[mask_p].var())
            else:
                score_matrix[i, j] = float(mag2[ay - half: ay + half, ax - half: ax + half].max())
    # The old intermediate step recomputed the aligned stack from the
    # float32-stored shifts.
    aligned_frames = np.empty((n_sel, H, W), dtype=np.float32)
    for i, frame in enumerate(frames):
        aligned_frames[i] = apply_shift(frame, float(global_shifts[i, 0]), float(global_shifts[i, 1]))
    return global_shifts, score_matrix, aligned_frames


# ── Synthetic data ────────────────────────────────────────────────────────────

SIZE = 160
CX = CY = 80.0


def _planet(ringed: bool) -> np.ndarray:
    """Oblate textured disk, optionally with a fainter Saturn-like ring —
    the ring makes find_disk_center take its core-isolation / radial-limb
    paths, which are the ones _gradient_disk_r serves."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    r = np.hypot(xx - CX, (yy - CY) / 0.93)
    img = np.zeros((SIZE, SIZE), np.float32)
    img[r < 45] = 0.7
    img += (0.12 * np.sin(yy / 3.5) * (r < 45)).astype(np.float32)
    if ringed:
        ring = np.abs(np.hypot(xx - CX, (yy - CY) * 3.2) - 66) < 7
        img[ring & (r >= 45)] = 0.35
    return cv2.GaussianBlur(img, (0, 0), 1.2)


def _frames(n: int, ringed: bool, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = _planet(ringed)
    out = []
    for _ in range(n):
        M = np.float32([[1, 0, rng.uniform(-3, 3)], [0, 1, rng.uniform(-3, 3)]])
        f = cv2.warpAffine(base, M, (SIZE, SIZE), flags=cv2.INTER_CUBIC)
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.5, 2.0))
        out.append(np.clip(f + rng.normal(0, 0.02, f.shape), 0, 1).astype(np.float32))
    return np.stack(out)


def _ap_positions():
    aps = [(ax, ay, 24) for ay in range(48, 113, 16) for ax in range(48, 113, 16)]
    # one larger AP, one centred off the disk (ring-feather branch in Pass 2),
    # one too close to the border to be scored (skipped branch)
    return aps + [(80, 80, 40), (30, 80, 24), (6, 6, 24)]


def _cfg(n_workers: int, **kw) -> LuckyStackConfig:
    base = dict(top_percent=0.5, ap_size=24, ap_grid_mode=APGridMode.AS4, use_ncc=True,
                per_ap_selection=True, n_ser_parallel=1, min_frames=5, n_workers=n_workers)
    base.update(kw)
    return LuckyStackConfig(**base)


class _ForceProcesses:
    """Make _per_ap_independent_stack use the process pool for tiny inputs."""
    def __enter__(self):
        self._old = ls._PASS1_PROCESS_MIN_FRAMES
        ls._PASS1_PROCESS_MIN_FRAMES = 0

    def __exit__(self, *_):
        ls._PASS1_PROCESS_MIN_FRAMES = self._old


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_subpixel_ray_edge_matches_reference():
    rng = np.random.default_rng(3)
    for n in (5, 9, 40, 100, 300):
        for _ in range(30):
            prof = np.cumsum(rng.normal(0, 1, n))[::-1] + rng.normal(0, 0.3, n)
            dr = float(rng.uniform(0.2, 2.0))
            for margin in (None, 0, 2, 5):
                assert derotation._subpixel_ray_edge(prof, dr, 1.5, margin) == \
                    _ref_subpixel_ray_edge(prof, dr, 1.5, margin)


def test_gradient_disk_r_matches_reference():
    rng = np.random.default_rng(4)
    n_cases = 0
    for ringed in (False, True):
        for f in _frames(6, ringed, seed=5):
            for search_frac in ((0.75, 1.30), (0.6, 1.8)):
                # r_rough up to 110 pushes many rays past the image border
                # (skipped rays) and some cases under the 8-valid-ray floor.
                for r_rough in (20.0, 45.0, 60.0, 110.0):
                    cx = CX + rng.uniform(-4, 4)
                    cy = CY + rng.uniform(-4, 4)
                    got = derotation._gradient_disk_r(f, cx, cy, r_rough,
                                                      search_frac=search_frac, return_n_valid=True)
                    ref = _ref_gradient_disk_r(f, cx, cy, r_rough,
                                               search_frac=search_frac, return_n_valid=True)
                    assert got == ref, (got, ref)
                    n_cases += 1
    noise = np.random.default_rng(6).random((SIZE, SIZE)).astype(np.float32)
    assert derotation._gradient_disk_r(noise, CX, CY, 40.0, return_n_valid=True) == \
        _ref_gradient_disk_r(noise, CX, CY, 40.0, return_n_valid=True)
    assert n_cases == 96


def test_batch_qsf_refine_matches_single():
    rng = np.random.default_rng(7)
    cc = rng.normal(0, 1, (25, 24, 24)).astype(np.float32)
    cc[:, 3, 20] += 8.0          # clear peaks, incl. wrap-around neighbours
    cc[5, 0, 0] += 12.0
    cc[6, 23, 23] += 12.0
    dx, dy = ls._batch_qsf_refine(cc)
    for k in range(len(cc)):
        sx, sy, _ = ls._qsf_refine(cc[k])
        assert (dx[k], dy[k]) == (sx, sy)


def test_score_frames_log_disk_threaded_matches_reference():
    frames = _frames(40, ringed=True, seed=8)
    frames[7] = 0.0              # an all-black frame hits the mask.sum() < 50 branch
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "synth.ser")
        w = SERWriter(path, {"ColorID": 0, "PixelDepth": 8}, SIZE, SIZE)
        for f in frames:
            w.write_frame((f * 255).astype(np.uint8))
        w.close()
        with SERReader(path) as rd, np.errstate(invalid="ignore", divide="ignore"):
            ref = _ref_score_frames_log_disk(rd, _cfg(1))
            for nw in (1, 4):
                got = ls.score_frames_log_disk(rd, _cfg(nw))
                assert np.array_equal(got, ref, equal_nan=True), nw


def test_pass1_frame_and_process_pool_match_reference():
    for ringed in (False, True):
        for metric in ("log_disk", "local_gradient"):
            frames = _frames(20, ringed, seed=9)
            reference = frames[:8].mean(0)
            aps = _ap_positions()
            cfg = _cfg(2, score_metric=metric)
            ref_gs, ref_sm, ref_al = _ref_pass1(frames, reference, aps, CX, CY, 45.0, cfg)

            p = ls._Pass1Params(reference, aps, CX, CY, 45.0, cfg.ap_size, 0, metric,
                                3, 3.0, 0.25)
            for i, f in enumerate(frames):
                dx, dy, row, aligned = ls._pass1_frame(f, p)
                assert np.array_equal(np.float32([dx, dy]), ref_gs[i])
                assert np.array_equal(row, ref_sm[i])
                assert np.array_equal(aligned, ref_al[i])

            holder = []
            try:
                gs, sm, al = ls._pass1_in_processes(frames, p, 2, holder)
                assert np.array_equal(gs, ref_gs)
                assert np.array_equal(sm, ref_sm)
                assert np.array_equal(al, ref_al)
                del al
            finally:
                for shm in holder:
                    ls._release_shm(shm)


def test_per_ap_stack_process_pool_matches_in_process():
    frames = _frames(30, ringed=True, seed=10)
    scores = np.array([float(cv2.Laplacian(f, cv2.CV_32F).var()) for f in frames], np.float32)
    sel_idx = np.argsort(scores)[::-1][:24].copy()
    reference = frames[sel_idx[:8]].mean(0)
    aps = _ap_positions()
    cases = {
        "mono": (frames[sel_idx], {}),
        "bayer": (np.clip(frames[sel_idx] * 0.9, 0, 1),
                  dict(bayer_code=cv2.COLOR_BayerBG2RGB, pixel_scale=255.0)),
    }
    for name, (sel, extra) in cases.items():
        in_proc, _ = ls._per_ap_independent_stack(sel, sel_idx, scores, reference, CX, CY, 45.0,
                                                  aps, _cfg(1), **extra)
        with _ForceProcesses():
            pooled, _ = ls._per_ap_independent_stack(sel, sel_idx, scores, reference, CX, CY,
                                                     45.0, aps, _cfg(2), **extra)
        assert np.array_equal(in_proc, pooled), name


if __name__ == "__main__":
    test_subpixel_ray_edge_matches_reference()
    test_gradient_disk_r_matches_reference()
    test_batch_qsf_refine_matches_single()
    test_score_frames_log_disk_threaded_matches_reference()
    test_pass1_frame_and_process_pool_match_reference()
    test_per_ap_stack_process_pool_matches_in_process()
    print("All checks passed.")
