"""CH4 filter presence status label for the CV-based position refinement
checkbox (Step 4).

Informational only -- it does NOT enable/disable the checkbox. That
checkbox covers both satellite-body refinement (needs CH4; silently
skipped for a window/session when CH4 is absent) and shadow refinement
(works with any filter, no CH4 needed), so CH4 presence must not gate the
checkbox itself. DerotatePanel instead gates it on the satellite-composite
checkbox being on (composite_enabled), since the refinement code only runs
inside the satellite-composite pass.

CH4 presence depends on the currently loaded *input directory*, which can
change mid-session (the user re-runs steps 01-03 against different raw
data). Ch4StatusRow exposes `check()` to be re-run every time
DerotatePanel.load_session() sees a (possibly new) input_dir, cancelling
any previous in-flight check first so a stale directory's result can't
land after the input has already changed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QWidget


class _Ch4Thread(QThread):
    done = Signal(bool)

    def __init__(self, input_dir: Path, target: str, parent=None):
        super().__init__(parent)
        self._input_dir = input_dir
        self._target = target

    def run(self):
        try:
            from pipeline.modules.image_io import group_by_filter
            groups = group_by_filter(self._input_dir, target=self._target)
            self.done.emit("CH4" in groups)
        except Exception:
            self.done.emit(False)


class Ch4StatusRow(QWidget):
    """[checkbox] [status_label] composite widget.

    The label reports whether CH4 data is present for the current input
    directory (satellite-body refinement only fires when it is) -- the
    checkbox itself is never disabled by this widget.
    """

    def __init__(self, checkbox: QCheckBox, parent=None):
        super().__init__(parent)
        self._cb = checkbox

        self._lbl = QLabel()
        self._lbl.setStyleSheet("color: #888; font-size: 11px;")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        lay.addWidget(self._cb)
        lay.addWidget(self._lbl)
        lay.addStretch()

        self._thread: Optional[_Ch4Thread] = None
        self._set_label("ch4_status.no_input", "#888")

    def _set_label(self, key: str, color: str):
        from gui.i18n import S
        self._lbl.setText(S(key))
        self._lbl.setStyleSheet(f"color: {color}; font-size: 11px;")

    def check(self, input_dir: Optional[Path], target: str = "Jup") -> None:
        """Re-run the CH4-presence check against *input_dir* (or clear if
        it's missing). Cancels any previous in-flight check. Never touches
        the checkbox's enabled state."""
        if self._thread is not None and self._thread.isRunning():
            try:
                self._thread.done.disconnect(self._on_done)
            except (TypeError, RuntimeError):
                pass
            self._thread.quit()
            self._thread.wait(50)

        if not input_dir or not Path(input_dir).is_dir():
            self._set_label("ch4_status.no_input", "#888")
            return

        self._set_label("ch4_status.checking", "#888")
        self._thread = _Ch4Thread(Path(input_dir), target, parent=self)
        self._thread.done.connect(self._on_done)
        self._thread.start()

    def _on_done(self, present: bool):
        if present:
            self._set_label("ch4_status.found", "#4caf50")
        else:
            self._set_label("ch4_status.not_found", "#888")
