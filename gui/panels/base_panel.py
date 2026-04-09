"""Base class for all step panels."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QProgressBar,
                                QPushButton, QScrollArea, QSizePolicy,
                                QVBoxLayout, QWidget)

from gui.i18n import S


def _fmt_remaining(seconds: float) -> str:
    """Format remaining seconds as a human-readable Korean string."""
    if seconds < 5:
        return "거의 완료..."
    if seconds < 60:
        return f"약 {int(seconds)+1}초 남음"
    minutes = int(seconds / 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"약 {minutes}분 {secs:02d}초 남음"
    hours = minutes // 60
    mins = minutes % 60
    return f"약 {hours}시간 {mins}분 남음"


class BasePanel(QWidget):
    """Common layout for every step panel.

    Subclasses override:
      - ``build_form()``  → add widgets to self._form_layout (QVBoxLayout)
      - ``get_config_updates()`` → dict of values to merge into session
      - ``output_paths()`` → list[Path] of output files (unused in UI now)
    """

    run_requested = Signal(str)   # step_id

    STEP_ID    : str  = ""
    TITLE_KEY  : str  = ""
    DESC_KEY   : str  = ""
    OPTIONAL   : bool = False
    HAS_NEXT   : bool = True

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._status = "idle"
        self._run_start_time: float | None = None
        self._build_skeleton()
        self.build_form()

    # ── Skeleton (shared across all steps) ────────────────────────────────────

    def _build_skeleton(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        # Header
        header = QHBoxLayout()
        step_num = QLabel(f"Step {self.STEP_ID}")
        step_num.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        step_num.setStyleSheet("color: #888;")
        header.addWidget(step_num)

        title = QLabel(S(self.TITLE_KEY) if self.TITLE_KEY else "")
        title.setFont(QFont("Arial", 13, QFont.Weight.Bold))
        title.setStyleSheet("color: #e8e8e8;")
        header.addWidget(title)
        header.addStretch()

        self._status_badge = QLabel(S("step.status.idle"))
        self._status_badge.setFixedWidth(70)
        self._status_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_badge.setStyleSheet(
            "background: #333; color: #888; border-radius: 10px; padding: 2px 6px;"
        )
        header.addWidget(self._status_badge)
        root.addLayout(header)

        # Description
        if self.DESC_KEY:
            desc = QLabel(S(self.DESC_KEY))
            desc.setWordWrap(True)
            desc.setStyleSheet("color: #999; font-size: 11px;")
            root.addWidget(desc)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #444;")
        root.addWidget(line)

        # Form area (scrollable)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        form_container = QWidget()
        self._form_layout = QVBoxLayout(form_container)
        self._form_layout.setContentsMargins(0, 0, 0, 0)
        self._form_layout.setSpacing(8)
        self._form_layout.addStretch()
        scroll.setWidget(form_container)
        root.addWidget(scroll, 1)

        # Action buttons
        btn_row = QHBoxLayout()
        self._btn_run = QPushButton(S("btn.run"))
        self._btn_run.setFixedHeight(32)
        self._btn_run.setStyleSheet(
            "QPushButton { background: #2d6a4f; color: white; border-radius: 5px; font-weight: bold; }"
            "QPushButton:hover { background: #40916c; }"
            "QPushButton:disabled { background: #333; color: #666; }"
        )
        self._btn_run.clicked.connect(lambda: self.run_requested.emit(self.STEP_ID))
        btn_row.addWidget(self._btn_run)

        self._btn_next = QPushButton(S("btn.next"))
        self._btn_next.setFixedHeight(32)
        self._btn_next.setStyleSheet(
            "QPushButton { background: #1a4a6e; color: white; border-radius: 5px; }"
            "QPushButton:hover { background: #2d6a9f; }"
        )
        btn_row.addWidget(self._btn_next)
        if not self.HAS_NEXT:
            self._btn_next.hide()
        btn_row.addStretch()
        root.addLayout(btn_row)

        # Progress bar (indeterminate by default, switches to determinate)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)   # busy / indeterminate
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedHeight(4)
        self._progress_bar.setStyleSheet(
            "QProgressBar { background: #2d2d2d; border: none; border-radius: 2px;"
            " color: #cccccc; font-size: 11px; }"
            "QProgressBar::chunk { background: #4da6ff; border-radius: 2px; }"
        )
        self._progress_bar.hide()
        root.addWidget(self._progress_bar)

        # Remaining time label (shown below progress bar when determinate)
        self._time_label = QLabel("")
        self._time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._time_label.setStyleSheet("color: #888; font-size: 10px;")
        self._time_label.hide()
        root.addWidget(self._time_label)

    # ── Subclass interface ─────────────────────────────────────────────────────

    def build_form(self) -> None:
        """Override to add config widgets into self._form_layout."""

    def get_config_updates(self) -> dict[str, Any]:
        """Override to return {session_key: value} for saving."""
        return {}

    def output_paths(self) -> list[Path]:
        """Override to return output file paths (informational, not displayed)."""
        return []

    # ── Status ─────────────────────────────────────────────────────────────────

    def set_status(self, status: str) -> None:
        self._status = status
        key  = f"step.status.{status}"
        text = S(key)
        colours = {
            "idle":    ("background: #333; color: #888;",),
            "running": ("background: #1a3a6e; color: #4da6ff;",),
            "success": ("background: #1a3a2a; color: #5cad5c;",),
            "error":   ("background: #4a1a1a; color: #e05c5c;",),
            "waiting": ("background: #3a2a1a; color: #e08c30;",),
            "skipped": ("background: #333; color: #555;",),
        }
        style = colours.get(status, ("background: #333; color: #888;",))[0]
        self._status_badge.setText(text)
        self._status_badge.setStyleSheet(
            f"{style} border-radius: 10px; padding: 2px 6px;"
        )

    def set_running(self, running: bool) -> None:
        self._btn_run.setEnabled(not running)
        self._btn_next.setEnabled(not running)
        if running:
            self._run_start_time = time.monotonic()
            # Start in indeterminate mode; set_progress() switches to determinate
            self._progress_bar.setRange(0, 0)
            self._progress_bar.setTextVisible(False)
            self._progress_bar.setFixedHeight(4)
            self._progress_bar.show()
            self._time_label.setText("")
        else:
            self._run_start_time = None
            self._progress_bar.hide()
            self._time_label.hide()
            # Reset for next run
            self._progress_bar.setRange(0, 0)
            self._progress_bar.setValue(0)
            self._progress_bar.setTextVisible(False)
            self._progress_bar.setFixedHeight(4)

    def set_progress(self, current: int, total: int) -> None:
        """Switch progress bar to determinate mode and update value."""
        if total > 0:
            self._progress_bar.setRange(0, total)
            self._progress_bar.setValue(current)
            self._progress_bar.setFormat(f"{current} / {total}")
            self._progress_bar.setTextVisible(True)
            self._progress_bar.setFixedHeight(16)
            # Remaining time estimate
            if self._run_start_time is not None and current > 0:
                elapsed = time.monotonic() - self._run_start_time
                remaining = elapsed * (total - current) / current
                self._time_label.setText(_fmt_remaining(remaining))
                self._time_label.show()
        else:
            self._progress_bar.setRange(0, 0)
            self._progress_bar.setTextVisible(False)
            self._progress_bar.setFixedHeight(4)

    def refresh_after_run(self) -> None:
        """Hook called after a successful run. Override in subclasses as needed."""

    # ── Helpers for subclasses ─────────────────────────────────────────────────

    @staticmethod
    def _add_separator(layout: QVBoxLayout) -> None:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #444;")
        layout.addWidget(line)
