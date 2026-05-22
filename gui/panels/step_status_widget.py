"""Shared status-dot widgets for step dependency display.

StepStatusWidget  — row of colored dots (● Step N) for auto-assigned panels.
FolderStatusDot   — single dot placed beside a folder path input for manual panels.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

_COL_READY   = "#4CAF50"   # green  — step output found
_COL_MISSING = "#e05252"   # red    — output_dir set but step not done
_COL_UNKNOWN = "#666666"   # gray   — output_dir not configured yet

_STEP_NAMES = {
    1: "SER Crop",
    2: "Lucky Stack",
    3: "Quality",
    4: "Derotate",
    5: "Wavelet Master",
    6: "RGB Composite",
    7: "Wavelet Preview",
    8: "GIF",
    9: "Summary Grid",
}


def check_step_ready(output_dir: Path, step_num: int) -> bool:
    """Return True if step N's output appears to exist under output_dir."""
    try:
        match step_num:
            case 1:
                d = output_dir / "step01_ser_crop"
                return d.exists() and next(d.iterdir(), None) is not None
            case 2:
                d = output_dir / "step02_lucky_stack"
                return next(d.rglob("*.tif"), None) is not None
            case 3:
                return (output_dir / "step03_quality" / "windows.json").exists()
            case 4:
                d = output_dir / "step04_derotated"
                return next(d.glob("*/*.tif"), None) is not None
            case 5:
                d = output_dir / "step05_wavelet_master"
                return next(d.glob("*/*.png"), None) is not None
            case 6:
                d = output_dir / "step06_rgb_composite"
                return next(d.glob("*/*.png"), None) is not None
            case 7:
                d = output_dir / "step07_wavelet_preview"
                return next(d.glob("*.png"), None) is not None
            case 8:
                d = output_dir / "step08_gif"
                return next(d.glob("*.gif"), None) is not None
            case 9:
                d = output_dir / "step09_summary_grid"
                return next(d.glob("*.png"), None) is not None
    except Exception:
        pass
    return False


class StepStatusWidget(QWidget):
    """Row of colored dots showing readiness of required upstream steps.

    Usage::
        widget = StepStatusWidget(steps=[3, 5])
        widget.refresh(output_dir)   # call on load and on output_dir change
    """

    def __init__(self, steps: list[int], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self._steps = steps
        self._dots: dict[int, QLabel] = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(0)

        for i, step_num in enumerate(steps):
            if i > 0:
                sep = QLabel("  ·  ")
                sep.setStyleSheet("color: #555; font-size: 11px;")
                layout.addWidget(sep)

            dot = QLabel("●")
            dot.setStyleSheet(f"color: {_COL_UNKNOWN}; font-size: 11px;")
            name = _STEP_NAMES.get(step_num, str(step_num))
            dot.setToolTip(name)

            lbl = QLabel(f" Step {step_num}")
            lbl.setStyleSheet("color: #999; font-size: 11px;")
            lbl.setToolTip(name)

            layout.addWidget(dot)
            layout.addWidget(lbl)
            self._dots[step_num] = dot

        layout.addStretch()

    def refresh(self, output_dir: Path | None) -> None:
        """Recheck each step and update dot colors."""
        for step_num, dot in self._dots.items():
            if output_dir is None:
                color = _COL_UNKNOWN
            elif check_step_ready(output_dir, step_num):
                color = _COL_READY
            else:
                color = _COL_MISSING
            dot.setStyleSheet(f"color: {color}; font-size: 11px;")


class FolderStatusDot(QLabel):
    """Single dot placed to the left of a folder path QLineEdit.

    Turns green when the folder contains at least one file matching *any*
    of the given glob patterns; red if the folder exists but is empty/wrong;
    gray if the path field is empty.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("●", parent)
        self.setStyleSheet(f"color: {_COL_UNKNOWN}; font-size: 11px;")

    def check(self, folder_path: str, patterns: list[str]) -> None:
        p = Path(folder_path) if folder_path.strip() else None
        if p is None:
            color = _COL_UNKNOWN
        elif not p.exists():
            color = _COL_MISSING
        elif any(next(p.glob(pat), None) is not None for pat in patterns):
            color = _COL_READY
        else:
            color = _COL_MISSING
        self.setStyleSheet(f"color: {color}; font-size: 11px;")
