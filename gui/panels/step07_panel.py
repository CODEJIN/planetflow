"""Step 7 — RGB composite (master) panel."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from gui.i18n import S
from gui.panels.base_panel import BasePanel
from gui.widgets.rgb_composite_preview import RgbCompositePreviewWidget

_SPINBOX_STYLE = (
    "QDoubleSpinBox { background: #3c3c3c; color: #d4d4d4; border: 1px solid #555;"
    " border-radius: 3px; padding: 3px 6px; }"
    "QDoubleSpinBox:focus { border-color: #4da6ff; }"
)
_READONLY_STYLE = (
    "QLineEdit { background: #2a2a2a; color: #888; border: 1px solid #3a3a3a;"
    " border-radius: 3px; padding: 3px 6px; }"
)
_COMBO_STYLE = (
    "QComboBox { background: #3c3c3c; color: #d4d4d4; border: 1px solid #555;"
    " border-radius: 3px; padding: 2px 6px; }"
    "QComboBox::drop-down { border: none; width: 20px; }"
    "QComboBox QAbstractItemView { background: #3c3c3c; color: #d4d4d4;"
    " selection-background-color: #4da6ff; }"
)
_NAME_STYLE = (
    "QLineEdit { background: #3c3c3c; color: #d4d4d4; border: 1px solid #555;"
    " border-radius: 3px; padding: 3px 6px; }"
    "QLineEdit:focus { border-color: #4da6ff; }"
)
_BTN_SMALL = (
    "QPushButton { background: #3c3c3c; color: #aaa; border: 1px solid #555;"
    " border-radius: 3px; padding: 2px 8px; font-size: 11px; }"
    "QPushButton:hover { background: #4a4a4a; color: #d4d4d4; }"
)
_BTN_ADD = (
    "QPushButton { background: #2d6a4f; color: white; border: none;"
    " border-radius: 3px; padding: 3px 10px; font-size: 11px; }"
    "QPushButton:hover { background: #40916c; }"
)

# Default specs shown when no session data is present
_DEFAULT_SPECS = [
    {"name": "RGB",      "R": "R",   "G": "G", "B": "B",  "L": ""},
    {"name": "IR-RGB",   "R": "R",   "G": "G", "B": "B",  "L": "IR"},   # LRGB: IR as luminance
    {"name": "CH4-G-IR", "R": "CH4", "G": "G", "B": "IR", "L": ""},
]


class _SpecRow(QWidget):
    """A single composite spec row: name + R/G/B/L dropdowns + delete button."""

    def __init__(
        self,
        filters: list[str],
        spec: dict[str, str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._filters = filters

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 1, 0, 1)
        row.setSpacing(4)

        self._radio = QRadioButton()
        self._radio.setFixedWidth(20)
        self._radio.setToolTip("이 합성을 미리보기에 표시")
        row.addWidget(self._radio)

        self._name = QLineEdit()
        self._name.setStyleSheet(_NAME_STYLE)
        self._name.setFixedWidth(90)
        self._name.setPlaceholderText("이름")
        row.addWidget(self._name)

        self._combos: dict[str, QComboBox] = {}
        for ch_label, ch_key in [("R", "R"), ("G", "G"), ("B", "B"), ("L", "L")]:
            lbl = QLabel(ch_label + ":")
            lbl.setStyleSheet("color: #aaa; font-size: 11px;")
            lbl.setFixedWidth(14)
            row.addWidget(lbl)

            cb = QComboBox()
            cb.setStyleSheet(_COMBO_STYLE)
            cb.setFixedWidth(70)
            # L channel can be empty (None)
            options = (["──"] if ch_key == "L" else []) + filters
            for opt in options:
                cb.addItem(opt)
            self._combos[ch_key] = cb
            row.addWidget(cb)

        self._btn_del = QPushButton("✕")
        self._btn_del.setStyleSheet(_BTN_SMALL)
        self._btn_del.setFixedWidth(24)
        self._btn_del.setToolTip("이 합성 설정 삭제")
        row.addWidget(self._btn_del)

        if spec:
            self._load(spec)

    def _load(self, spec: dict[str, str]) -> None:
        self._name.setText(spec.get("name", ""))
        for ch in ("R", "G", "B", "L"):
            val = spec.get(ch, "")
            cb = self._combos[ch]
            idx = cb.findText(val if val else "──")
            if idx >= 0:
                cb.setCurrentIndex(idx)

    def update_filters(self, filters: list[str]) -> None:
        """Rebuild combo items when the global filter list changes."""
        self._filters = filters
        for ch, cb in self._combos.items():
            current = cb.currentText()
            cb.clear()
            options = (["──"] if ch == "L" else []) + filters
            for opt in options:
                cb.addItem(opt)
            idx = cb.findText(current)
            cb.setCurrentIndex(max(0, idx))

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self._name.text().strip(),
            "R": self._combos["R"].currentText(),
            "G": self._combos["G"].currentText(),
            "B": self._combos["B"].currentText(),
            "L": "" if self._combos["L"].currentText() == "──" else self._combos["L"].currentText(),
        }


class Step07Panel(BasePanel):
    STEP_ID   = "07"
    TITLE_KEY = "step07.title"
    DESC_KEY  = "step07.desc"
    OPTIONAL  = False

    def __init__(self, parent: QWidget | None = None) -> None:
        self._output_dir: Path | None = None
        self._spec_rows: list[_SpecRow] = []
        self._available_filters: list[str] = ["R", "G", "B", "IR", "CH4", "UV"]
        super().__init__(parent)

    # ── BasePanel interface ───────────────────────────────────────────────────

    def build_form(self) -> None:
        # ── Horizontal split: controls (left) | preview (right) ────────────
        main_widget = QWidget()
        main_widget.setStyleSheet("background: transparent;")
        main_hlayout = QHBoxLayout(main_widget)
        main_hlayout.setSpacing(16)
        main_hlayout.setContentsMargins(0, 0, 0, 0)

        # ── Left: controls ──────────────────────────────────────────────────
        left_widget = QWidget()
        left_widget.setStyleSheet("background: transparent;")
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(8)
        left_layout.setContentsMargins(0, 0, 0, 0)

        form_widget = QWidget()
        form_widget.setStyleSheet("background: transparent;")
        fl = QFormLayout(form_widget)
        fl.setSpacing(10)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        # Auto-derived folder display
        self._input_lbl = QLineEdit()
        self._input_lbl.setReadOnly(True)
        self._input_lbl.setStyleSheet(_READONLY_STYLE)
        self._input_lbl.setToolTip("Step 6 마스터 이미지가 있는 폴더 (자동)")
        lbl_in = QLabel(S("step07.input_dir"))
        lbl_in.setToolTip("Step 6에서 출력된 마스터 PNG 파일이 있는 폴더입니다.")
        fl.addRow(lbl_in, self._input_lbl)

        self._output_lbl = QLineEdit()
        self._output_lbl.setReadOnly(True)
        self._output_lbl.setStyleSheet(_READONLY_STYLE)
        lbl_out = QLabel(S("step07.output_dir"))
        lbl_out.setToolTip("RGB 합성 결과물이 저장될 폴더입니다.")
        fl.addRow(lbl_out, self._output_lbl)

        # Max shift px
        _tip_shift = (
            "채널 간 정렬 시 허용되는 최대 이동량(px)입니다.\n"
            "너무 크면 잘못된 채널이 정렬될 수 있습니다."
        )
        self._max_shift = QDoubleSpinBox()
        self._max_shift.setStyleSheet(_SPINBOX_STYLE)
        self._max_shift.setRange(0.0, 100.0)
        self._max_shift.setDecimals(1)
        self._max_shift.setSingleStep(1.0)
        self._max_shift.setValue(15.0)
        self._max_shift.setToolTip(_tip_shift)
        lbl_shift = QLabel(S("step07.max_shift"))
        lbl_shift.setToolTip(_tip_shift)
        fl.addRow(lbl_shift, self._max_shift)

        left_layout.addWidget(form_widget)

        # ── Composite spec table ──────────────────────────────────────────────
        spec_header = QWidget()
        spec_header.setStyleSheet("background: transparent;")
        hdr_layout = QHBoxLayout(spec_header)
        hdr_layout.setContentsMargins(0, 8, 0, 4)
        hdr_layout.setSpacing(0)
        hdr_lbl = QLabel("합성 설정 (Composite Specs)")
        hdr_lbl.setStyleSheet("color: #aaa; font-size: 11px; font-weight: bold;")
        hdr_lbl.setToolTip(
            "각 합성 이미지의 이름과 채널 매핑을 정의합니다.\n"
            "R/G/B: 각 색 채널에 할당할 필터\n"
            "L: 루마(밝기) 채널 (LRGB 합성 시 사용, 선택적)"
        )
        hdr_layout.addWidget(hdr_lbl)
        hdr_layout.addStretch()
        self._btn_add_spec = QPushButton("+ 추가")
        self._btn_add_spec.setStyleSheet(_BTN_ADD)
        self._btn_add_spec.setToolTip("새 합성 설정 추가")
        self._btn_add_spec.clicked.connect(self._add_spec_row)
        hdr_layout.addWidget(self._btn_add_spec)

        left_layout.addWidget(spec_header)

        # Scrollable container for spec rows
        self._spec_container = QWidget()
        self._spec_container.setStyleSheet("background: #252525; border-radius: 4px;")
        self._spec_vbox = QVBoxLayout(self._spec_container)
        self._spec_vbox.setContentsMargins(6, 6, 6, 6)
        self._spec_vbox.setSpacing(2)

        # Column header
        col_hdr = QWidget()
        col_hdr.setStyleSheet("background: transparent;")
        col_hdr_layout = QHBoxLayout(col_hdr)
        col_hdr_layout.setContentsMargins(0, 0, 0, 2)
        col_hdr_layout.setSpacing(4)
        for txt, w in [("👁", 20), ("이름", 90), ("R채널", 84), ("G채널", 84), ("B채널", 84), ("L채널", 84), ("", 24)]:
            lbl = QLabel(txt)
            lbl.setFixedWidth(w)
            lbl.setStyleSheet("color: #666; font-size: 10px;")
            col_hdr_layout.addWidget(lbl)
        self._spec_vbox.addWidget(col_hdr)

        scroll = QScrollArea()
        scroll.setWidget(self._spec_container)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.Shape.NoFrame)
        scroll.setMaximumHeight(200)
        scroll.setStyleSheet("QScrollArea { background: #252525; border-radius: 4px; }")
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        left_layout.addWidget(scroll)
        left_layout.addStretch()
        main_hlayout.addWidget(left_widget, 1)

        # Radio button group — controls which spec is previewed
        self._spec_radio_group = QButtonGroup(self)
        self._spec_radio_group.buttonClicked.connect(self._on_params_changed)

        # Populate defaults (before preview, so _preview not yet set)
        for spec in _DEFAULT_SPECS:
            self._add_spec_row(spec)

        # ── Right: preview ──────────────────────────────────────────────────
        self._preview = RgbCompositePreviewWidget(parent=self)
        main_hlayout.addWidget(self._preview, 0)

        # Connect existing spec rows' combos to preview updates
        for row in self._spec_rows:
            for cb in row._combos.values():
                cb.currentIndexChanged.connect(self._on_params_changed)

        idx = self._form_layout.count() - 1
        self._form_layout.insertWidget(idx, main_widget)

    def _add_spec_row(self, spec: dict[str, str] | None = None) -> None:
        row = _SpecRow(self._available_filters, spec, parent=self._spec_container)
        row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row._btn_del.clicked.connect(lambda _, r=row: self._remove_spec_row(r))
        self._spec_rows.append(row)
        self._spec_vbox.addWidget(row)
        # Register radio button in the group; first row is always selected
        if hasattr(self, "_spec_radio_group"):
            self._spec_radio_group.addButton(row._radio)
            if len(self._spec_rows) == 1:
                row._radio.setChecked(True)
        # Connect combos to preview update (only when preview already exists)
        if hasattr(self, "_preview"):
            for cb in row._combos.values():
                cb.currentIndexChanged.connect(self._on_params_changed)

    def _remove_spec_row(self, row: _SpecRow) -> None:
        was_checked = row._radio.isChecked()
        if hasattr(self, "_spec_radio_group"):
            self._spec_radio_group.removeButton(row._radio)
        if row in self._spec_rows:
            self._spec_rows.remove(row)
        row.setParent(None)
        row.deleteLater()
        # If the removed row was the active preview, switch to first remaining
        if was_checked and self._spec_rows:
            self._spec_rows[0]._radio.setChecked(True)
            self._on_params_changed()

    def _refresh_filter_options(self, filters: list[str]) -> None:
        self._available_filters = filters
        for row in self._spec_rows:
            row.update_filters(filters)

    def get_config_updates(self) -> dict[str, Any]:
        specs = [r.to_dict() for r in self._spec_rows if r.to_dict()["name"]]
        return {
            "max_shift_px": self._max_shift.value(),
            "composite_specs": specs,
        }

    def load_session(self, data: dict[str, Any]) -> None:
        out = data.get("output_dir", "")
        if out:
            p = Path(out)
            self._input_lbl.setText(str(p / "step06_wavelet_master"))
            self._output_lbl.setText(str(p / "step07_rgb_composite"))
            if hasattr(self, "_preview"):
                self._preview.set_input_dir(p / "step06_wavelet_master")
        self._max_shift.setValue(float(data.get("max_shift_px", 15.0)))

        # Update filter list from settings
        raw = data.get("filters", "")
        if raw:
            filters = [f.strip() for f in raw.split(",") if f.strip()]
            if filters:
                self._refresh_filter_options(filters)

        # Reload specs if saved
        saved_specs = data.get("composite_specs")
        if saved_specs:
            # Clear existing rows
            for row in list(self._spec_rows):
                if hasattr(self, "_spec_radio_group"):
                    self._spec_radio_group.removeButton(row._radio)
                row.setParent(None)
                row.deleteLater()
            self._spec_rows.clear()
            for spec in saved_specs:
                self._add_spec_row(spec)

        # Sync preview spec from first row
        self._sync_preview_spec()

    def output_paths(self) -> list[Path]:
        if self._output_dir is None:
            return []
        step_dir = self._output_dir / "step07_rgb_composite"
        if not step_dir.exists():
            return []
        return sorted(step_dir.rglob("*.png"))

    def set_output_dir(self, path: Path | str) -> None:
        self._output_dir = Path(path) if path else None
        step06_dir = self._output_dir / "step06_wavelet_master" if self._output_dir else None
        if hasattr(self, "_preview"):
            self._preview.set_input_dir(step06_dir)

    # ── Qt events ─────────────────────────────────────────────────────────────

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if hasattr(self, "_preview"):
            self._preview.schedule_update(150)

    def refresh_after_run(self) -> None:
        super().refresh_after_run()
        if hasattr(self, "_preview"):
            self._preview.schedule_update(500)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_params_changed(self) -> None:
        if not hasattr(self, "_preview"):
            return
        self._sync_preview_spec()
        self._preview.schedule_update()

    def _sync_preview_spec(self) -> None:
        """Push the radio-selected spec row's channel mapping into the preview widget."""
        if not hasattr(self, "_preview") or not self._spec_rows:
            return
        active = next(
            (r for r in self._spec_rows if r._radio.isChecked()),
            self._spec_rows[0],
        )
        spec = active.to_dict()
        self._preview.set_spec(
            r_filter=spec.get("R", "R"),
            g_filter=spec.get("G", "G"),
            b_filter=spec.get("B", "B"),
            l_filter=spec.get("L", ""),
        )
