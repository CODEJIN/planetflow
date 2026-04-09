"""Step 2 — AutoStakkert 4 (external tool) panel.

This panel is informational only.  It explains what the user needs to do
with AS!4 (an external stacking tool) before proceeding to Step 3.
Emits ``completed`` when the user clicks the Continue button.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui.i18n import S

_PANEL_BG = "#252526"
_BTN_CONTINUE = (
    "QPushButton { background: #2d6a4f; color: white; border-radius: 5px;"
    " font-weight: bold; padding: 6px 20px; border: none; }"
    "QPushButton:hover { background: #40916c; }"
)
_WARNING_STYLE = (
    "background: #2a2200; border: 1px solid #665500; border-radius: 6px;"
    " padding: 12px;"
)


class Step02Panel(QWidget):
    """Informational panel for the AutoStakkert 4 manual step."""

    completed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        self.setStyleSheet(f"background: {_PANEL_BG};")
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(16)

        # Header
        header = QHBoxLayout()
        step_num = QLabel("Step 02")
        step_num.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        step_num.setStyleSheet("color: #888;")
        header.addWidget(step_num)

        title = QLabel(S("step02.title"))
        title.setFont(QFont("Arial", 13, QFont.Weight.Bold))
        title.setStyleSheet("color: #e8e8e8;")
        header.addWidget(title)
        header.addStretch()
        root.addLayout(header)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #444;")
        root.addWidget(line)

        # Warning / instruction box
        warn_box = QWidget()
        warn_box.setStyleSheet(_WARNING_STYLE)
        warn_layout = QVBoxLayout(warn_box)
        warn_layout.setContentsMargins(0, 0, 0, 0)
        warn_layout.setSpacing(8)

        warn_icon = QLabel("⚠  " + S("step02.warning_title"))
        warn_icon.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        warn_icon.setStyleSheet("color: #ffcc00; background: transparent; border: none;")
        warn_layout.addWidget(warn_icon)

        warn_text = QLabel(S("step02.instructions"))
        warn_text.setWordWrap(True)
        warn_text.setStyleSheet("color: #ccc; background: transparent; border: none; font-size: 12px;")
        warn_layout.addWidget(warn_text)

        root.addWidget(warn_box)

        # Hint
        hint = QLabel(S("step02.hint2"))
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #777; font-size: 11px; font-style: italic;")
        root.addWidget(hint)

        root.addStretch()

        # Continue button
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._btn_continue = QPushButton(S("btn.continue"))
        self._btn_continue.setFixedHeight(34)
        self._btn_continue.setMinimumWidth(220)
        self._btn_continue.setStyleSheet(_BTN_CONTINUE)
        self._btn_continue.clicked.connect(self.completed)
        btn_row.addWidget(self._btn_continue)
        root.addLayout(btn_row)

    # ── Public API (kept for compatibility) ───────────────────────────────────

    def update_paths(self, ser_dir, output_dir) -> None:
        """No-op: paths are now managed in step01/step03 panels."""

    def start_watching(self, output_dir) -> None:
        """No-op: folder watching removed."""

    def set_output_dir(self, path) -> None:
        """No-op."""
