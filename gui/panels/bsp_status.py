"""BSP kernel availability check widget for satellite composite panels.

Checks whether Skyfield BSP files (de440s.bsp, jup365.bsp) exist in the
standard location. If missing, tests internet connectivity to naif.jpl.nasa.gov
and updates the checkbox row accordingly.
"""
from __future__ import annotations

import os
import socket
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QWidget

# Mirror resolution order from satellite_tracker (no circular import)
_PRIMARY_DIR = Path(
    os.environ.get("PLANETFLOW_SKYFIELD_DIR", "")
    or Path.home() / ".planetflow" / "skyfield"
)
_LEGACY_DIR = Path("/tmp/skyfield")
_REQUIRED = [("de440s.bsp", "32 MB"), ("jup365.bsp", "1.1 GB")]


def _all_present(d: Path) -> bool:
    return all((d / name).exists() for name, _ in _REQUIRED)


def _bsp_available() -> bool:
    return _all_present(_PRIMARY_DIR) or _all_present(_LEGACY_DIR)


def _missing_from_primary() -> list[tuple[str, str]]:
    return [(n, s) for n, s in _REQUIRED if not (_PRIMARY_DIR / n).exists()]


def _internet_ok() -> bool:
    try:
        socket.create_connection(("naif.jpl.nasa.gov", 443), timeout=3)
        return True
    except OSError:
        return False


class _BspThread(QThread):
    done = Signal(str, str)  # ("ok" | "download" | "offline", detail)

    def run(self):
        try:
            import skyfield  # noqa: F401
        except ImportError:
            self.done.emit("no_skyfield", "")
            return
        if _bsp_available():
            self.done.emit("ok", "")
            return
        if _internet_ok():
            missing = _missing_from_primary()
            detail = " + ".join(f"{n} ({s})" for n, s in missing)
            self.done.emit("download", detail)
        else:
            self.done.emit("offline", "")


class BspStatusRow(QWidget):
    """[checkbox] [status_label] composite widget.

    The checkbox is disabled until the background BSP check completes.
    On result:
      ok       → green "OK", checkbox enabled
      download → orange "<files> — 사용 시 자동 다운로드", checkbox enabled
      offline  → red "네트워크 연결 필요", checkbox disabled
    """

    def __init__(self, checkbox: QCheckBox, parent=None):
        super().__init__(parent)
        self._cb = checkbox
        self._cb.setEnabled(False)

        self._lbl = QLabel()
        self._lbl.setStyleSheet("color: #888; font-size: 11px;")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        lay.addWidget(self._cb)
        lay.addWidget(self._lbl)
        lay.addStretch()

        self._thread = _BspThread()
        self._thread.done.connect(self._on_done)
        self._refresh_checking_label()
        self._thread.start()

    def _refresh_checking_label(self):
        from gui.i18n import S
        self._lbl.setText(S("bsp.checking"))

    def _on_done(self, status: str, detail: str):
        from gui.i18n import S
        if status == "ok":
            self._lbl.setText(S("bsp.ok"))
            self._lbl.setStyleSheet("color: #4caf50; font-size: 11px;")
            self._cb.setEnabled(True)
        elif status == "no_skyfield":
            self._lbl.setText(S("bsp.no_skyfield"))
            self._lbl.setStyleSheet("color: #f44336; font-size: 11px;")
            self._cb.setEnabled(False)
        elif status == "download":
            self._lbl.setText(f"{detail} — {S('bsp.auto_download')}")
            self._lbl.setStyleSheet("color: #ff9800; font-size: 11px;")
            self._cb.setEnabled(True)
        else:
            self._lbl.setText(S("bsp.offline"))
            self._lbl.setStyleSheet("color: #f44336; font-size: 11px;")
            self._cb.setEnabled(False)
