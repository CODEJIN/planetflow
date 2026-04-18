"""Welcome / home panel — shown on startup instead of Settings."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPainter
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui.i18n import S

# Logo path — works both in development and PyInstaller frozen builds
_HERE = Path(__file__).parent
_LOGO_PATH = str(
    (Path(sys._MEIPASS) / "gui" / "icons" / "logo_planetflow.svg")
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")
    else _HERE.parent / "icons" / "logo_planetflow.svg"
)

# Step IDs for the pipeline status display
_STEP_IDS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

_STATUS_COLOR = {
    "success": ("#2d6b30", "#4caf50", "#a5d6a7"),  # bg, border, text
    "error":   ("#6b2020", "#f44336", "#ef9a9a"),
    "running": ("#1a3a6b", "#2196f3", "#90caf9"),
    "skipped": ("#2a2a2a", "#555",    "#666"),
    "":        ("#282828", "#3c3c3c", "#555"),
}

class _SvgWidget(QWidget):
    """Transparent SVG renderer — no background fill, so the card surface shows through."""

    def __init__(self, path: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._renderer = QSvgRenderer(path, self)
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._renderer.render(p)
        p.end()


_BTN_PRIMARY = (
    "QPushButton { background: #2d5a1b; color: #b8f5a0; border: 1px solid #4a9030;"
    " border-radius: 5px; padding: 8px 20px; font-size: 13px; font-weight: bold; }"
    "QPushButton:hover { background: #3a7a25; }"
    "QPushButton:pressed { background: #1e3d12; }"
)
_BTN_SECONDARY = (
    "QPushButton { background: #3c3c3c; color: #bbb; border: 1px solid #555;"
    " border-radius: 5px; padding: 8px 20px; font-size: 13px; }"
    "QPushButton:hover { background: #4a4a4a; color: #e0e0e0; }"
)


class WelcomePanel(QWidget):
    """Landing screen displayed on app startup."""

    go_settings = Signal()
    go_resume   = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: #1e1e1e;")
        self._step_dots: dict[str, QLabel] = {}
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.setContentsMargins(0, 0, 0, 0)

        # Centre card
        card = QWidget()
        card.setFixedWidth(620)
        card.setStyleSheet(
            "background: #252526; border: 1px solid #3c3c3c; border-radius: 8px;"
        )
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(16)
        card_layout.setContentsMargins(40, 28, 40, 28)

        # PlanetFlow logo (SVG, 4:1 aspect → 512×128)
        logo = _SvgWidget(_LOGO_PATH)
        logo.setFixedSize(512, 128)
        card_layout.addWidget(logo, alignment=Qt.AlignmentFlag.AlignCenter)

        # Subtitle
        self._subtitle_lbl = QLabel(S("welcome.subtitle"))
        self._subtitle_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._subtitle_lbl.setStyleSheet(
            "color: #666; font-size: 11px; background: transparent; border: none;"
        )
        card_layout.addWidget(self._subtitle_lbl)

        card_layout.addWidget(_sep())

        # Pipeline status label
        self._status_header = QLabel(S("welcome.pipeline_status"))
        self._status_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_header.setStyleSheet(
            "color: #888; font-size: 11px; background: transparent; border: none;"
        )
        card_layout.addWidget(self._status_header)

        # Step dots — 2 rows × 5
        dots_widget = QWidget()
        dots_widget.setStyleSheet("background: transparent;")
        grid = QGridLayout(dots_widget)
        grid.setSpacing(6)
        grid.setContentsMargins(0, 0, 0, 0)

        for i, step_id in enumerate(_STEP_IDS):
            dot = QLabel(S(f"step.short.{step_id}"))
            dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            dot.setFixedSize(86, 42)
            dot.setWordWrap(False)
            dot.setStyleSheet(_dot_style(""))
            dot.setFont(QFont("Arial", 8))
            self._step_dots[step_id] = dot
            grid.addWidget(dot, i // 5, i % 5)

        card_layout.addWidget(dots_widget, alignment=Qt.AlignmentFlag.AlignCenter)

        # Legend
        legend = QLabel(
            f'<span style="color:#4caf50">●</span> {S("welcome.legend.done")}'
            f'　<span style="color:#f44336">●</span> {S("welcome.legend.error")}'
            f'　<span style="color:#3c3c3c">●</span> {S("welcome.legend.idle")}'
        )
        legend.setAlignment(Qt.AlignmentFlag.AlignCenter)
        legend.setStyleSheet(
            "color: #555; font-size: 10px; background: transparent; border: none;"
        )
        card_layout.addWidget(legend)

        card_layout.addWidget(_sep())

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        btn_row.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._btn_settings = QPushButton("⚙  " + S("app.settings"))
        self._btn_settings.setFixedSize(170, 40)
        self._btn_settings.setStyleSheet(_BTN_SECONDARY)
        self._btn_settings.clicked.connect(self.go_settings)
        btn_row.addWidget(self._btn_settings)

        self._btn_resume = QPushButton("▶  " + S("welcome.btn_resume"))
        self._btn_resume.setFixedSize(170, 40)
        self._btn_resume.setStyleSheet(_BTN_PRIMARY)
        self._btn_resume.clicked.connect(self.go_resume)
        btn_row.addWidget(self._btn_resume)

        card_layout.addLayout(btn_row)

        outer.addWidget(card, alignment=Qt.AlignmentFlag.AlignCenter)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_session(self, data: dict[str, Any]) -> None:
        """Update step status dots from session data."""
        step_status = data.get("step_status", {})
        for step_id, dot in self._step_dots.items():
            status = step_status.get(step_id, "")
            dot.setStyleSheet(_dot_style(status))

    def retranslate(self) -> None:
        """Re-apply i18n strings after language change."""
        self._subtitle_lbl.setText(S("welcome.subtitle"))
        self._status_header.setText(S("welcome.pipeline_status"))
        self._btn_settings.setText("⚙  " + S("app.settings"))
        self._btn_resume.setText("▶  " + S("welcome.btn_resume"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet("color: #3c3c3c; background: #3c3c3c; border: none;")
    f.setFixedHeight(1)
    return f


def _dot_style(status: str) -> str:
    bg, border, text = _STATUS_COLOR.get(status, _STATUS_COLOR[""])
    return (
        f"background: {bg}; color: {text}; border: 1px solid {border};"
        f" border-radius: 4px; font-size: 8px;"
    )
