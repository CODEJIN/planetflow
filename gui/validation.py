"""Pre-flight validation infrastructure for pipeline steps.

Each step panel overrides ``BasePanel.validate(config)`` and returns a list
of ``ValidationIssue`` objects.  An empty list means the step is ready to run.

Design principles:
- Adding a new check = appending one item to the list in the relevant panel.
  No architecture change needed.
- severity="error"   → blocks execution (shown in red)
- severity="warning" → non-blocking advisory (shown in yellow)
- Validation is always O(1) or O(file_count): no file I/O beyond directory
  listing, no image reads.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ValidationIssue:
    severity: str   # "error" | "warning"
    message: str


# ── Shared helpers ─────────────────────────────────────────────────────────────

def count_files(folder: str, *patterns: str) -> int:
    """Return number of files matching any of the glob patterns in *folder*."""
    if not folder:
        return 0
    p = Path(folder)
    if not p.exists():
        return 0
    return sum(len(list(p.glob(pat))) for pat in patterns)


def filter_files_in_dir(folder: str, filter_name: str, extensions=("*.tif", "*.TIF")) -> int:
    """Count TIF files whose stem contains *filter_name* (case-insensitive)."""
    if not folder:
        return 0
    p = Path(folder)
    if not p.exists():
        return 0
    total = 0
    low = filter_name.lower()
    for ext in extensions:
        for f in p.glob(ext):
            if low in f.stem.lower():
                total += 1
    return total
