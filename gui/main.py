"""Application entry point for the planetary imaging pipeline GUI."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so both 'gui' and 'pipeline' are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QColor, QCursor, QFont, QIcon, QPalette
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QComboBox,
                                QLineEdit, QSlider, QToolTip, QWidget)

_ICONS_DIR = Path(__file__).parent / "icons"

from gui import session
from gui import i18n
from gui.main_window import MainWindow


def _build_dark_palette() -> QPalette:
    """Return a QPalette matching the dark #1e1e1e theme."""
    p = QPalette()
    bg      = QColor("#1e1e1e")
    panel   = QColor("#252526")
    text    = QColor("#d4d4d4")
    mid     = QColor("#3c3c3c")
    link    = QColor("#4da6ff")
    disabl  = QColor("#666666")

    p.setColor(QPalette.ColorRole.Window,          bg)
    p.setColor(QPalette.ColorRole.WindowText,      text)
    p.setColor(QPalette.ColorRole.Base,            panel)
    p.setColor(QPalette.ColorRole.AlternateBase,   bg)
    p.setColor(QPalette.ColorRole.ToolTipBase,     QColor("#2d2d2d"))
    p.setColor(QPalette.ColorRole.ToolTipText,     text)
    p.setColor(QPalette.ColorRole.Text,            text)
    p.setColor(QPalette.ColorRole.Button,          mid)
    p.setColor(QPalette.ColorRole.ButtonText,      text)
    p.setColor(QPalette.ColorRole.BrightText,      QColor("#ffffff"))
    p.setColor(QPalette.ColorRole.Link,            link)
    p.setColor(QPalette.ColorRole.Highlight,       QColor("#264f78"))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabl)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, disabl)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabl)
    return p


class AstroApp(QApplication):
    """QApplication subclass that force-shows tooltips on Linux/Fusion/dark theme.

    PySide6 on Linux with Fusion style and a dark palette silently suppresses
    tooltip display.  Overriding notify() is more reliable than installEventFilter
    because notify() is guaranteed to fire for every event delivered to any
    object — unlike installEventFilter which can be silently bypassed in some
    PySide6 builds.

    The parent-widget traversal handles composite widgets (QSpinBox, QDoubleSpinBox,
    etc.) whose internal QLineEdit receives the ToolTip event instead of the outer
    widget where setToolTip() was actually called.
    """

    def notify(self, obj: QObject, event: QEvent) -> bool:
        # Suppress scroll-wheel value changes on spin boxes, combo boxes and
        # sliders unless the widget is explicitly focused (user clicked into it).
        # Also covers the internal QLineEdit inside a QAbstractSpinBox, which
        # receives wheel events when hovering over the text portion of the spinbox.
        if event.type() == QEvent.Type.Wheel and isinstance(obj, QWidget):
            target = obj
            # Unwrap spinbox's internal QLineEdit → treat as the spinbox itself
            if isinstance(obj, QLineEdit):
                parent = obj.parentWidget()
                if isinstance(parent, QAbstractSpinBox):
                    target = parent
            if isinstance(target, (QAbstractSpinBox, QComboBox, QSlider)):
                if not target.hasFocus():
                    event.ignore()
                    return True

        if event.type() == QEvent.Type.ToolTip and isinstance(obj, QWidget):
            w: QWidget | None = obj
            while w is not None:
                tip = w.toolTip()
                if tip:
                    QToolTip.showText(QCursor.pos(), tip, w)  # type: ignore[arg-type]
                    return True
                w = w.parentWidget()
            # No tooltip found: hide any currently visible tooltip and let Qt handle it
            QToolTip.hideText()
            return True
        return super().notify(obj, event)


def main() -> None:
    # Load session to get language preference
    sess = session.load()
    lang = sess.get("language", "ko")

    # Initialise i18n before any widget is created
    i18n.load(lang)

    app = AstroApp(sys.argv)
    app.setApplicationName("AstroPipeline")
    app.setOrganizationName("AstroImaging")
    app.setPalette(_build_dark_palette())
    app.setStyle("Fusion")   # Fusion base style lets our dark palette take full effect

    # Set application icon
    _icon_path = _ICONS_DIR / "app_icon.svg"
    if _icon_path.exists():
        app.setWindowIcon(QIcon(str(_icon_path)))

    # Set QToolTip stylesheet at the *application* level.
    # QToolTip is a top-level popup window — widget-level stylesheets (set on
    # QMainWindow) do not affect it.  This must come after setStyle().
    app.setStyleSheet(
        "QToolTip {"
        "  background-color: #2d2d2d;"
        "  color: #d4d4d4;"
        "  border: 1px solid #666;"
        "  border-radius: 4px;"
        "  padding: 6px 8px;"
        "  font-size: 11px;"
        "}"
    )

    # Explicitly set the tooltip font so it's readable independent of platform
    tip_font = QFont()
    tip_font.setPointSize(11)
    QToolTip.setFont(tip_font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
