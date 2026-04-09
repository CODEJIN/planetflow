# -*- mode: python ; coding: utf-8 -*-
"""
AstroPipeline PyInstaller spec — 플랫폼 공통 (Linux / Windows)

사용법:
  Linux   : pyinstaller --clean astro_pipeline.spec
  Windows : pyinstaller --clean astro_pipeline.spec
"""
import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_submodules,
)

block_cipher = None
IS_WINDOWS = sys.platform == "win32"

# ── Linux: libexpat 명시적 번들 ──────────────────────────────────────────────
# Python 3.13의 pyexpat.so는 libexpat >= 2.6.0 심볼을 요구하지만,
# 시스템 libexpat(Ubuntu 22.04: 2.4.7)이 더 오래됐을 수 있음.
# PyInstaller가 conda/venv의 libexpat을 "시스템 라이브러리"로 간주해 제외하므로
# 빌드 환경(sys.prefix)의 libexpat.so.1을 수동으로 번들에 포함.
_extra_binaries = []
if not IS_WINDOWS:
    _libexpat = Path(sys.prefix) / "lib" / "libexpat.so.1"
    if _libexpat.exists():
        _extra_binaries.append((str(_libexpat), "."))

# ── PySide6: 플러그인 포함 전체 수집 ──────────────────────────────────────────
pyside6_datas, pyside6_binaries, pyside6_hiddenimports = collect_all("PySide6")
# astroquery도 데이터 파일이 많으므로 전체 수집 권장
aq_datas, aq_binaries, aq_hidden = collect_all("astroquery")

# ── 과학 라이브러리 서브모듈 ────────────────────────────────────────────────
scipy_hidden   = collect_submodules("scipy")
skimage_hidden = collect_submodules("skimage")
astropy_datas  = collect_data_files("astropy")

a = Analysis(
    ["gui/main.py"],
    pathex=["."],           # 프로젝트 루트를 검색 경로에 추가
    binaries=pyside6_binaries + _extra_binaries,
    datas=[
        # ── 앱 에셋 ────────────────────────────────────────────────────────
        ("gui/icons",     "gui/icons"),      # SVG 아이콘
        ("gui/i18n",      "gui/i18n"),       # 언어 JSON (ko/en)
        ("pipeline/data", "pipeline/data"),  # np_ang_table.json 등
        # ── 라이브러리 데이터 ───────────────────────────────────────────────
        *pyside6_datas,
        *aq_datas,
        *astropy_datas,
    ],
    hiddenimports=[
        # OpenCV
        "cv2",
        # 이미지 포맷
        "tifffile",
        "imageio",
        "imageio_ffmpeg",
        "PIL",
        "PIL.Image",
        "PIL.ImageFont",
        "PIL.ImageDraw",
        # scipy — PyInstaller가 동적 import를 놓치는 경우가 많음
        *scipy_hidden,
        # scikit-image
        *skimage_hidden,
        # astropy / astroquery
        "astropy",
        "erfa", # astropy의 필수 의존성
        *aq_hidden,
        # PySide6
        *pyside6_hiddenimports,
        # 사용자가 언급한 'gui' 패키지 인식 보완
        "gui",
        "pipeline",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["build_runtime_hook.py"] if os.path.exists("build_runtime_hook.py") else [],
    excludes=[
        # GUI 앱에서 불필요한 무거운 패키지 제거
        "tkinter",
        "matplotlib",
        "notebook",
        "jupyter",
        "IPython",
        "sphinx",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="AstroPipeline",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX 압축: Qt 관련 바이너리에서 충돌 가능성 있음 — 안전하게 비활성화
    upx=False,
    runtime_tmpdir=None,
    # console=False → 터미널 창 없이 GUI만 표시
    # 빌드/디버그 중에는 True로 바꾸면 오류 메시지를 볼 수 있음
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # 아이콘: Windows는 .ico, Linux는 .png 사용
    # ICO 파일을 준비한 뒤 아래 주석을 해제하세요
    # icon="gui/icons/app_icon.ico",  # Windows
    # icon="gui/icons/app_icon.png",  # Linux (선택적)
)
