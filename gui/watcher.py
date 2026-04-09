"""Folder watcher for Step 2 — monitors AS!4 output directory.

Uses Qt's QFileSystemWatcher so no extra thread is required.
Emits tif_count_changed(int) whenever the number of .tif files changes.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QFileSystemWatcher, QObject, Signal


class FolderWatcher(QObject):
    """Watches a directory and emits the current .tif file count on change."""

    tif_count_changed = Signal(int)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._watcher = QFileSystemWatcher(self)
        self._watcher.directoryChanged.connect(self._on_changed)
        self._folder: Path | None = None

    def watch(self, folder: str | Path) -> None:
        """Start watching *folder*.  Removes previous watch if any."""
        folder = Path(folder)
        if self._folder and str(self._folder) in self._watcher.directories():
            self._watcher.removePath(str(self._folder))
        self._folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        self._watcher.addPath(str(folder))
        # Emit current count immediately
        self._on_changed(str(folder))

    def stop(self) -> None:
        if self._folder:
            self._watcher.removePath(str(self._folder))
            self._folder = None

    def _on_changed(self, _path: str) -> None:
        if self._folder is None:
            return
        count = len(list(self._folder.glob("*.tif")))
        self.tif_count_changed.emit(count)
