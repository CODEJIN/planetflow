"""Base class for all step panels."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QProgressBar,
                                QPushButton, QScrollArea, QSizePolicy,
                                QVBoxLayout, QWidget)

from gui.i18n import S


def _fmt_remaining(seconds: float) -> str:
    """Format remaining seconds as a human-readable string."""
    if seconds < 5:
        return S("fmt.almost_done")
    if seconds < 60:
        return S("fmt.seconds_left", n=int(seconds) + 1)
    minutes = int(seconds / 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return S("fmt.minutes_left", m=minutes, s=secs)
    hours = minutes // 60
    mins = minutes % 60
    return S("fmt.hours_left", h=hours, m=mins)


class BasePanel(QWidget):
    """Common layout for every step panel.

    Subclasses override:
      - ``build_form()``  → add widgets to self._form_layout (QVBoxLayout)
      - ``get_config_updates()`` → dict of values to merge into session
      - ``output_paths()`` → list[Path] of output files (unused in UI now)
    """

    run_requested  = Signal(str)   # step_id
    stop_requested = Signal()      # user clicked Stop

    STEP_ID    : str  = ""
    TITLE_KEY  : str  = ""
    DESC_KEY   : str  = ""
    OPTIONAL   : bool = False
    HAS_NEXT   : bool = True

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._status = "idle"
        self._run_start_time: float | None = None
        self._cancelling = False
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

        self._btn_stop = QPushButton(S("btn.stop"))
        self._btn_stop.setFixedHeight(32)
        self._btn_stop.setStyleSheet(
            "QPushButton { background: #6a2d2d; color: white; border-radius: 5px; font-weight: bold; }"
            "QPushButton:hover { background: #9c4040; }"
            "QPushButton:disabled { background: #3a2a2a; color: #777; }"
        )
        self._btn_stop.clicked.connect(self._on_stop_clicked)
        self._btn_stop.hide()
        btn_row.addWidget(self._btn_stop)

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

    def validate(self, config: dict, batch_mode: bool = False) -> list:
        """Pre-flight checks for this step.

        Override in subclasses to add validation logic.  Return a list of
        ``gui.validation.ValidationIssue`` objects.  An empty list means the
        step is ready to run.

        Add new checks by appending items to the list — no architecture change
        needed.  Example::

            from gui.validation import ValidationIssue, count_files
            issues = []
            if not count_files(config.get("input_dir", ""), "*.tif", "*.TIF"):
                issues.append(ValidationIssue("error", S("validate.no_tif_input")))
            return issues
        """
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
        if running:
            self._cancelling = False
            self._run_start_time = time.monotonic()
            self._btn_run.setEnabled(False)
            self._btn_next.setEnabled(False)
            self._btn_stop.setText(S("btn.stop"))
            self._btn_stop.setEnabled(True)
            self._btn_stop.setStyleSheet(
                "QPushButton { background: #6a2d2d; color: white; border-radius: 5px; font-weight: bold; }"
                "QPushButton:hover { background: #9c4040; }"
                "QPushButton:disabled { background: #3a2a2a; color: #777; }"
            )
            self._btn_stop.show()
            self._progress_bar.setRange(0, 0)
            self._progress_bar.setTextVisible(False)
            self._progress_bar.setFixedHeight(4)
            self._progress_bar.show()
            self._time_label.setText("")
        else:
            # Always clean up progress bar
            self._run_start_time = None
            self._progress_bar.hide()
            self._time_label.hide()
            self._progress_bar.setRange(0, 0)
            self._progress_bar.setValue(0)
            self._progress_bar.setTextVisible(False)
            self._progress_bar.setFixedHeight(4)
            if not self._cancelling:
                # Normal completion: re-enable controls, hide stop button
                self._btn_run.setEnabled(True)
                self._btn_next.setEnabled(True)
                self._btn_stop.hide()

    def _on_stop_clicked(self) -> None:
        self._cancelling = True
        self._btn_stop.setText(S("btn.stopping"))
        self._btn_stop.setEnabled(False)
        self.stop_requested.emit()

    def on_cancelled(self) -> None:
        """Called when the runner has truly finished all threads after a stop request."""
        self._cancelling = False
        self._btn_stop.setText(S("btn.stopped"))
        self._btn_stop.setStyleSheet(
            "QPushButton { background: #1a4a2d; color: #5cff88; border: 1px solid #5cff88;"
            " border-radius: 5px; font-weight: bold; }"
        )
        self._btn_stop.setEnabled(False)
        self._btn_stop.show()
        self.set_status("idle")
        QTimer.singleShot(2500, self._reset_after_cancel)

    def _reset_after_cancel(self) -> None:
        self._btn_stop.hide()
        self._btn_run.setEnabled(True)
        self._btn_next.setEnabled(True)

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
