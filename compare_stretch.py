"""
글로벌 필터 정규화 stretch 방식 비교.
개별 프레임 TIF → derotation 없이 단순 평균 스택 → stretch 비교.

방식 A (현재): 프레임별 Pass1[0.5%~99.5%] 정규화 → 스택 → compose[0.1%~99.9%]
방식 B (수정): 프레임 스택 → Pass1[0.1%~99.9%] 한 번만
방식 C (기준): 프레임 스택 → compose[0.1%~99.9%] (글로벌 정규화 OFF)

결과물: /data/astro_test/stretch_compare/
"""
import sys
sys.path.insert(0, "/data/astro_test")

import numpy as np
from pathlib import Path
from PIL import Image
from pipeline.modules import image_io

OUT_DIR = Path("/data/astro_test/stretch_compare")
OUT_DIR.mkdir(exist_ok=True)
RAW_DIR = Path("/data/astro_test/260407")

# 필터별 TIF 파일 수집
def collect_frames():
    frames = {"R": [], "G": [], "B": []}
    for p in sorted(RAW_DIR.glob("*.tif")):
        n = p.stem
        for f in frames:
            tag = f"-U-{f}-Jup"
            if tag in n:
                frames[f].append(p)
    for f, paths in frames.items():
        print(f"  {f}: {len(paths)}개 프레임")
    return frames

def load_stack(paths) -> np.ndarray:
    """단순 평균 스택 (derotation 없이)."""
    imgs = []
    for p in paths:
        img = image_io.read_tif(p)
        if img.ndim == 3:
            img = img.mean(axis=2).astype(np.float32)
        imgs.append(img)
    return np.mean(imgs, axis=0).astype(np.float32)

def auto_stretch(img, plow, phigh):
    lo = float(np.percentile(img, plow))
    hi = float(np.percentile(img, phigh))
    span = hi - lo if hi > lo else 1.0
    return np.clip((img - lo) / span, 0.0, 1.0).astype(np.float32)

def to_rgb8(r, g, b):
    return (np.clip(np.stack([r, g, b], axis=2), 0, 1) * 255).astype(np.uint8)

def save_with_label(rgb8, path, label):
    img = Image.fromarray(rgb8)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 20], fill=(0, 0, 0))
    draw.text((4, 3), label, fill=(255, 255, 0))
    img.save(path)
    print(f"  → {path}")


print("프레임 수집 중...")
frames = collect_frames()

# ── 방식 A: 프레임별 정규화 → 스택 → compose ──────────────────────────────────
print("\n방식 A: 프레임별 Pass1[0.5-99.5] → 스택 → compose[0.1-99.9]")
# 글로벌 lo/hi 계산 (모든 RGB 프레임 합산)
all_pix = []
raw_stacks_A = {}
for f in ["R", "G", "B"]:
    for p in frames[f]:
        img = image_io.read_tif(p)
        if img.ndim == 3:
            img = img.mean(axis=2).astype(np.float32)
        all_pix.append(img.ravel())
combined = np.concatenate(all_pix)
lo1A = float(np.percentile(combined, 0.5))
hi1A = float(np.percentile(combined, 99.5))
span1A = hi1A - lo1A
print(f"  Pass1 global: lo={lo1A:.5f}  hi={hi1A:.5f}")

for f in ["R", "G", "B"]:
    normed_frames = []
    for p in frames[f]:
        img = image_io.read_tif(p)
        if img.ndim == 3:
            img = img.mean(axis=2).astype(np.float32)
        normed = np.clip((img - lo1A) / span1A, 0.0, 1.0).astype(np.float32)
        normed_frames.append(normed)
    raw_stacks_A[f] = np.mean(normed_frames, axis=0).astype(np.float32)

# compose stretch on already-normed stack
combined2 = np.concatenate([raw_stacks_A[f].ravel() for f in ["R","G","B"]])
lo2A = float(np.percentile(combined2, 0.1))
hi2A = float(np.percentile(combined2, 99.9))
span2A = hi2A - lo2A if hi2A > lo2A else 1.0
print(f"  compose: lo={lo2A:.5f}  hi={hi2A:.5f}")

outA = {f: np.clip((raw_stacks_A[f] - lo2A) / span2A, 0.0, 1.0) for f in ["R","G","B"]}
save_with_label(to_rgb8(outA["R"], outA["G"], outA["B"]),
                OUT_DIR / "A_frame_norm_then_stack.png",
                "A: 프레임별 정규화→스택→compose (현재)")

# ── 방식 B: 스택 → 단일 stretch ───────────────────────────────────────────────
print("\n방식 B: 스택 → Pass1[0.1-99.9] 한 번만")
raw_stacks_B = {f: load_stack(frames[f]) for f in ["R","G","B"]}
combined_B = np.concatenate([raw_stacks_B[f].ravel() for f in ["R","G","B"]])
lo1B = float(np.percentile(combined_B, 0.1))
hi1B = float(np.percentile(combined_B, 99.9))
span1B = hi1B - lo1B
print(f"  Pass1: lo={lo1B:.5f}  hi={hi1B:.5f}")
outB = {f: np.clip((raw_stacks_B[f] - lo1B) / span1B, 0.0, 1.0) for f in ["R","G","B"]}
save_with_label(to_rgb8(outB["R"], outB["G"], outB["B"]),
                OUT_DIR / "B_stack_then_stretch.png",
                "B: 스택→단일 stretch (수정안)")

# ── 방식 C: 글로벌 정규화 없이 compose만 (B와 동일 계산이나 명시적으로 분리) ────
print("\n방식 C: compose[0.1-99.9] only (글로벌 정규화 OFF와 동일)")
outC = {f: auto_stretch(raw_stacks_B[f], 0.1, 99.9) for f in ["R","G","B"]}
# joint로 다시 계산
combined_C = np.concatenate([raw_stacks_B[f].ravel() for f in ["R","G","B"]])
lo1C = float(np.percentile(combined_C, 0.1))
hi1C = float(np.percentile(combined_C, 99.9))
print(f"  compose: lo={lo1C:.5f}  hi={hi1C:.5f}")
outC = {f: np.clip((raw_stacks_B[f] - lo1C) / (hi1C - lo1C), 0.0, 1.0) for f in ["R","G","B"]}
save_with_label(to_rgb8(outC["R"], outC["G"], outC["B"]),
                OUT_DIR / "C_no_global_norm.png",
                "C: 글로벌 정규화 OFF")

# ── 나란히 비교 ────────────────────────────────────────────────────────────────
print("\n나란히 비교 이미지 생성")
pA = Image.open(OUT_DIR / "A_frame_norm_then_stack.png")
pB = Image.open(OUT_DIR / "B_stack_then_stretch.png")
pC = Image.open(OUT_DIR / "C_no_global_norm.png")
gap = 6
w, h = pA.width, pA.height
canvas = Image.new("RGB", (w*3 + gap*2, h), (30, 30, 30))
canvas.paste(pA, (0, 0))
canvas.paste(pB, (w + gap, 0))
canvas.paste(pC, (w*2 + gap*2, 0))
canvas.save(OUT_DIR / "compare_side_by_side.png")
print(f"  → {OUT_DIR / 'compare_side_by_side.png'}")
print("\n완료.")
