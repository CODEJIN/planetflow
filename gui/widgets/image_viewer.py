"""Single-image viewer with slider navigation for step result PNGs."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QDialog, QFrame, QHBoxLayout, QLabel, QSlider,
                                QSizePolicy, QVBoxLayout, QWidget)

_DISPLAY_H = 300   # image display area height in pixels


class ImageViewer(QWidget):
    """Shows one result image at a time with a slider to navigate between files.

    Design rationale:
    - Loading all thumbnails at once wastes memory and clutters the UI when
      steps produce dozens of PNGs.
    - A single large image view + slider gives better image quality, less
      visual noise, and lets the user scan results frame-by-frame.
    - Images are loaded on demand (one at a time) as the slider moves.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._paths: list[Path] = []
        self._current: int = 0
        self._build_ui()
        self.hide()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)

        # Image display area
        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_label.setMinimumHeight(_DISPLAY_H)
        self._img_label.setMaximumHeight(_DISPLAY_H)
        self._img_label.setSizePolicy(QSizePolicy.Policy.Expanding,
                                      QSizePolicy.Policy.Fixed)
        self._img_label.setStyleSheet(
            "background: #111; border-radius: 4px; border: 1px solid #3a3a3a;"
        )
        self._img_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self._img_label.setToolTip("클릭하면 원본 크기로 봅니다")
        self._img_label.mousePressEvent = lambda _e: self._open_full()
        layout.addWidget(self._img_label)

        # Navigation row
        nav = QWidget()
        nav.setStyleSheet("background: transparent;")
        nav_layout = QHBoxLayout(nav)
        nav_layout.setContentsMargins(2, 0, 2, 0)
        nav_layout.setSpacing(8)

        self._name_lbl = QLabel("")
        self._name_lbl.setStyleSheet("color: #888; font-size: 10px;")
        self._name_lbl.setSizePolicy(QSizePolicy.Policy.Expanding,
                                     QSizePolicy.Policy.Preferred)
        nav_layout.addWidget(self._name_lbl)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self._slider.setStyleSheet(
            "QSlider::groove:horizontal { background: #3a3a3a; height: 4px;"
            " border-radius: 2px; }"
            "QSlider::handle:horizontal { background: #4da6ff; width: 12px; height: 12px;"
            " margin: -4px 0; border-radius: 6px; }"
            "QSlider::sub-page:horizontal { background: #4da6ff; border-radius: 2px; }"
        )
        self._slider.valueChanged.connect(self._on_slider_changed)
        nav_layout.addWidget(self._slider, 1)

        self._count_lbl = QLabel("")
        self._count_lbl.setStyleSheet("color: #666; font-size: 10px;")
        self._count_lbl.setFixedWidth(46)
        self._count_lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        nav_layout.addWidget(self._count_lbl)

        layout.addWidget(nav)

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_paths(self, paths: list[Path]) -> None:
        """Load a list of image paths. Hides the widget if no valid paths."""
        self._paths = [p for p in paths if p.exists()]
        if not self._paths:
            self.hide()
            return

        self._slider.blockSignals(True)
        self._slider.setMinimum(0)
        self._slider.setMaximum(len(self._paths) - 1)
        self._slider.setValue(0)
        self._slider.blockSignals(False)

        self._current = 0
        self._update_display()
        self.show()

    def is_empty(self) -> bool:
        return not self._paths

    # ── Internals ──────────────────────────────────────────────────────────────

    def _on_slider_changed(self, value: int) -> None:
        self._current = value
        self._update_display()

    def _update_display(self) -> None:
        if not self._paths:
            return
        p = self._paths[self._current]
        self._name_lbl.setText(p.name)
        self._count_lbl.setText(f"{self._current + 1}/{len(self._paths)}")

        px = QPixmap(str(p))
        if not px.isNull():
            # Scale to fit display area (height-constrained, full-width target)
            w = max(self._img_label.width(), 600)
            px = px.scaled(w, _DISPLAY_H,
                           Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
        self._img_label.setPixmap(px)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        # Re-scale the current image when the widget is resized
        if self._paths:
            self._update_display()

    def _open_full(self) -> None:
        if not self._paths:
            return
        p = self._paths[self._current]
        dlg = QDialog(self)
        dlg.setWindowTitle(p.name)
        dlg.setStyleSheet("background: #1a1a1a;")
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(8, 8, 8, 8)
        label = QLabel()
        px = QPixmap(str(p))
        if not px.isNull():
            px = px.scaled(1200, 900,
                           Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
        label.setPixmap(px)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        dlg.exec()
