"""Colour-coded log viewer widget."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (QHBoxLayout, QPlainTextEdit, QPushButton,
                                QVBoxLayout, QWidget)

from gui.i18n import S

# Colour rules: (substring_match, fg_colour)
_RULES: list[tuple[str, str]] = [
    ("===",     "#5dd4f0"),   # section headers → cyan
    ("[ERROR]", "#ff5c5c"),   # errors → red
    ("[WARN]",  "#ffc965"),   # warnings → orange
    ("WARNING", "#ffc965"),
    ("→",       "#a8e6a3"),   # saved files → light green
    ("✓",       "#a8e6a3"),
    ("done",    "#a8e6a3"),
]
_DEFAULT_FG = "#d4d4d4"
_BG         = "#1e1e1e"
_FONT_FAMILY = "Consolas, Fira Mono, monospace"
_FONT_SIZE   = 10


def _colour_for(line: str) -> str:
    for pattern, colour in _RULES:
        if pattern in line:
            return colour
    return _DEFAULT_FG


class LogWidget(QWidget):
    """Scrollable, colour-coded, read-only log viewer."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    # ── Construction ──────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        # Toolbar
        toolbar = QHBoxLayout()
        self._btn_clear = QPushButton(S("log.clear"))
        self._btn_clear.setFixedWidth(60)
        self._btn_clear.clicked.connect(self.clear)
        self._btn_copy = QPushButton(S("log.copy"))
        self._btn_copy.setFixedWidth(60)
        self._btn_copy.clicked.connect(self._copy_all)
        toolbar.addStretch()
        toolbar.addWidget(self._btn_clear)
        toolbar.addWidget(self._btn_copy)
        root.addLayout(toolbar)

        # Text area
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(5000)
        self._text.setFont(QFont(_FONT_FAMILY, _FONT_SIZE))
        self._text.setStyleSheet(
            f"QPlainTextEdit {{ background: {_BG}; color: {_DEFAULT_FG};"
            f" border: 1px solid #444; border-radius: 4px; }}"
        )
        root.addWidget(self._text)

    # ── Public API ────────────────────────────────────────────────────────────

    def retranslate(self) -> None:
        self._btn_clear.setText(S("log.clear"))
        self._btn_copy.setText(S("log.copy"))

    def append_line(self, line: str) -> None:
        """Append one log line with colour-coding.  Thread-safe via Qt signal."""
        colour = _colour_for(line)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(colour))

        cursor = self._text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(line.rstrip("\n") + "\n", fmt)

        # Auto-scroll to bottom
        sb = self._text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear(self) -> None:
        self._text.clear()

    def _copy_all(self) -> None:
        self._text.selectAll()
        self._text.copy()
        cursor = self._text.textCursor()
        cursor.clearSelection()
        self._text.setTextCursor(cursor)
