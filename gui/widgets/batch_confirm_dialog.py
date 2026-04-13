"""Batch execution confirmation dialog — graphical pipeline flow.

Replaces the plain-text QDialog with a visual pipeline diagram showing all
steps as labeled nodes connected by arrows.  Hover each node to see details
(output path, validation issues).  Steps with validation errors are shown in
red and block the Run button.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# ── Step metadata ─────────────────────────────────────────────────────────────

# (step_id, short_name, long_name) — must match _STEP_DEFS in main_window.py
_STEP_META = [
    ("01", "PIPP",      "PIPP 전처리"),
    ("02", "Lucky\nStack", "Lucky Stacking"),
    ("03", "품질\n평가",   "품질 평가"),
    ("04", "드로\n테이션", "De-rotation 스태킹"),
    ("05", "웨이\n블릿",   "웨이블릿 마스터"),
    ("06", "RGB\n합성",   "RGB 합성"),
    ("07", "미리\n보기",  "웨이블릿 미리보기"),
    ("08", "시계\n열",    "시계열 RGB 합성"),
    ("09", "GIF",        "애니메이션 GIF"),
    ("10", "요약\n그리드", "요약 그리드"),
]

# ── Node colors ───────────────────────────────────────────────────────────────

_COLOR_RUN     = ("#2d6b30", "#4caf50", "#c8e6c9")   # bg, border, text
_COLOR_SKIP    = ("#282828", "#444",    "#555")
_COLOR_ERROR   = ("#6b2020", "#f44336", "#ef9a9a")
_COLOR_WARNING = ("#5a4a10", "#ffc107", "#fff59d")


# ── Validation issue type (matches gui/validation.py) ─────────────────────────

@dataclass
class _Issue:
    severity: str  # "error" | "warning"
    message: str


# ── StepNode widget ───────────────────────────────────────────────────────────

class _StepNode(QWidget):
    """Compact node in the pipeline diagram."""

    def __init__(
        self,
        step_id: str,
        short_name: str,
        long_name: str,
        state: str,          # "run" | "skip"
        output_path: str,
        issues: list[_Issue],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._step_id    = step_id
        self._long_name  = long_name
        self._state      = state
        self._out        = output_path
        self._issues     = issues
        self.setFixedSize(68, 82)
        self._build_ui(short_name)
        self._apply_style()
        self._build_tooltip()

    def _build_ui(self, short_name: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 4)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self._dot = QLabel("●")
        self._dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._dot.setFont(QFont("Arial", 14))
        layout.addWidget(self._dot)

        self._name_lbl = QLabel(short_name)
        self._name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._name_lbl.setFont(QFont("Arial", 8))
        self._name_lbl.setWordWrap(True)
        layout.addWidget(self._name_lbl)

        self._id_lbl = QLabel(f"Step {self._step_id}")
        self._id_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._id_lbl.setFont(QFont("Arial", 7))
        layout.addWidget(self._id_lbl)

    def _apply_style(self) -> None:
        has_error   = any(i.severity == "error"   for i in self._issues)
        has_warning = any(i.severity == "warning" for i in self._issues)

        if self._state == "skip":
            bg, border, fg = _COLOR_SKIP
        elif has_error:
            bg, border, fg = _COLOR_ERROR
        elif has_warning:
            bg, border, fg = _COLOR_WARNING
        else:
            bg, border, fg = _COLOR_RUN

        self.setStyleSheet(
            f"QWidget {{ background: {bg}; border: 1px solid {border};"
            f" border-radius: 5px; }}"
        )
        self._dot.setStyleSheet(f"color: {border}; background: transparent; border: none;")
        self._name_lbl.setStyleSheet(f"color: {fg}; background: transparent; border: none;")
        self._id_lbl.setStyleSheet(f"color: {border}; background: transparent; border: none;")

    def _build_tooltip(self) -> None:
        lines: list[str] = [self._long_name]
        if self._state == "skip":
            lines.append("(건너뜀)")
        else:
            if self._out:
                lines.append(f"→ {self._out}")
            for issue in self._issues:
                icon = "⛔" if issue.severity == "error" else "⚠"
                lines.append(f"{icon} {issue.message}")
        self.setToolTip("\n".join(lines))


# ── Dialog ────────────────────────────────────────────────────────────────────

class BatchConfirmDialog(QDialog):
    """Graphical pipeline confirmation dialog.

    Parameters
    ----------
    steps:        ordered list of step_ids that will actually run
    all_defs:     full _STEP_DEFS list (step_id, label_key, optional)
    start_from:   first step_id in the batch
    output_paths: mapping step_id → output path string
    input_summary:pre-built "입력: path (SER × N)" string
    issues:       optional {step_id: [ValidationIssue]} from pre-flight checks
    """

    def __init__(
        self,
        parent: QWidget | None,
        steps: list[str],
        all_defs: list,
        start_from: str,
        output_paths: dict[str, str],
        input_summary: str,
        issues: dict[str, list] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("일괄 실행 확인")
        self.setModal(True)
        self.setMinimumWidth(820)

        self._steps        = set(steps)
        self._all_defs     = all_defs
        self._start_from   = start_from
        self._output_paths = output_paths
        self._issues       = issues or {}

        self._has_errors = any(
            any(i.severity == "error" for i in lst)
            for lst in self._issues.values()
        )

        self._build_ui(input_summary)

    def _build_ui(self, input_summary: str) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(20, 16, 20, 14)

        # Input summary
        inp_lbl = QLabel(input_summary)
        inp_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(inp_lbl)

        layout.addWidget(_hline())

        # Pipeline flow
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent;")
        scroll.setFixedHeight(110)

        flow_widget = QWidget()
        flow_widget.setStyleSheet("background: transparent;")
        flow_layout = QHBoxLayout(flow_widget)
        flow_layout.setContentsMargins(4, 4, 4, 4)
        flow_layout.setSpacing(0)
        flow_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)

        all_ids = [sid for sid, _, _ in self._all_defs]
        start_idx = all_ids.index(self._start_from)

        for i, (step_id, _, optional) in enumerate(self._all_defs):
            if i < start_idx:
                continue

            short, long = _step_names(step_id)
            state  = "run" if step_id in self._steps else "skip"
            out    = self._output_paths.get(step_id, "")
            raw_issues = self._issues.get(step_id, [])
            # Wrap raw issues (may be ValidationIssue from gui.validation or bare _Issue)
            wrapped = [_Issue(getattr(x, "severity", "error"), getattr(x, "message", str(x))) for x in raw_issues]

            node = _StepNode(step_id, short, long, state, out, wrapped)
            flow_layout.addWidget(node)

            if i < len(self._all_defs) - 1:
                arrow = QLabel("→")
                arrow.setFixedWidth(16)
                arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
                arrow.setStyleSheet("color: #555; font-size: 12px; background: transparent;")
                flow_layout.addWidget(arrow)

        flow_layout.addStretch()
        scroll.setWidget(flow_widget)
        layout.addWidget(scroll)

        # Legend
        legend = QLabel(
            '<span style="color:#4caf50">●</span> 실행 예정'
            '　<span style="color:#444">●</span> 건너뜀'
            '　<span style="color:#f44336">●</span> 오류 (실행 불가)'
            '　<span style="color:#ffc107">●</span> 경고'
        )
        legend.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(legend)

        if self._has_errors:
            err_lbl = QLabel("⛔  검증 오류가 있어 실행할 수 없습니다. 해당 스텝 설정을 확인해주세요.")
            err_lbl.setStyleSheet(
                "color: #ef9a9a; font-size: 11px;"
                " background: #3a1010; border: 1px solid #f44336;"
                " border-radius: 4px; padding: 6px 10px;"
            )
            layout.addWidget(err_lbl)

        layout.addWidget(_hline())

        # Buttons
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok_btn = btns.button(QDialogButtonBox.StandardButton.Ok)
        ok_btn.setText("실행")
        ok_btn.setEnabled(not self._has_errors)
        if self._has_errors:
            ok_btn.setStyleSheet("color: #555; background: #2a2a2a;")
        else:
            ok_btn.setStyleSheet(
                "QPushButton { background: #2d5a1b; color: #b8f5a0;"
                " border: 1px solid #4a9030; border-radius: 4px; padding: 4px 16px; }"
                "QPushButton:hover { background: #3a7a25; }"
            )
        btns.button(QDialogButtonBox.StandardButton.Cancel).setText("취소")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)


# ── Helpers ───────────────────────────────────────────────────────────────────

_META_LOOKUP = {sid: (short, long) for sid, short, long in _STEP_META}


def _step_names(step_id: str) -> tuple[str, str]:
    return _META_LOOKUP.get(step_id, (step_id, f"Step {step_id}"))


def _hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet("color: #3c3c3c; background: #3c3c3c; border: none;")
    f.setFixedHeight(1)
    return f
