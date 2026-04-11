"""Thumbnail image grid widget for displaying step output PNGs."""
from __future__ import annotations

from pathlib import Path

from gui.i18n import S

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QDialog, QFrame, QHBoxLayout, QLabel,
                                QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

_THUMB_SIZE = 240   # thumbnail square size in pixels
_MAX_SHOWN  = 30    # max thumbnails before "show more" label


class _ThumbnailLabel(QLabel):
    clicked = Signal(Path)

    def __init__(self, path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._path = path
        px = QPixmap(str(path))
        if not px.isNull():
            px = px.scaled(_THUMB_SIZE, _THUMB_SIZE,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(px)
        self.setFixedSize(_THUMB_SIZE, _THUMB_SIZE)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("border: 1px solid #555; border-radius: 3px; background: #111;")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(path.name)

    def mousePressEvent(self, event):  # noqa: N802
        self.clicked.emit(self._path)
        super().mousePressEvent(event)


class ImageGrid(QWidget):
    """Horizontally scrollable thumbnail strip — shown at the bottom of each step panel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Height = thumbnail + scrollbar space
        self._scroll.setFixedHeight(_THUMB_SIZE + 22)
        self._scroll.setStyleSheet(
            "QScrollArea { background: #1a1a1a; border-radius: 4px; }"
            "QScrollBar:horizontal { background: #2d2d2d; height: 8px; }"
            "QScrollBar::handle:horizontal { background: #555; border-radius: 4px; min-width: 20px; }"
        )
        layout.addWidget(self._scroll)

        self._container = QWidget()
        self._container.setStyleSheet("background: #1a1a1a;")
        self._hbox = QHBoxLayout(self._container)
        self._hbox.setSpacing(6)
        self._hbox.setContentsMargins(6, 4, 6, 4)
        self._hbox.addStretch()
        self._scroll.setWidget(self._container)

        # Hide by default until images are loaded
        self.hide()

    def load_paths(self, paths: list[Path]) -> None:
        """Replace thumbnails with images from *paths*."""
        # Clear existing (everything before the trailing stretch)
        while self._hbox.count() > 1:
            item = self._hbox.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        shown = [p for p in paths[:_MAX_SHOWN] if p.exists()]

        if not shown:
            self.hide()
            return

        for p in shown:
            thumb = _ThumbnailLabel(p)
            thumb.clicked.connect(self._show_full)
            self._hbox.insertWidget(self._hbox.count() - 1, thumb)

        if len(paths) > _MAX_SHOWN:
            more = QLabel(S("image_grid.more", n=len(paths) - _MAX_SHOWN))
            more.setStyleSheet("color: #888; padding: 8px;")
            self._hbox.insertWidget(self._hbox.count() - 1, more)

        self.show()

    def _show_full(self, path: Path) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(path.name)
        layout = QVBoxLayout(dlg)
        label  = QLabel()
        px = QPixmap(str(path))
        # Scale to at most 900×900
        if not px.isNull():
            px = px.scaled(900, 900,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
        label.setPixmap(px)
        layout.addWidget(label)
        dlg.exec()
