"""Step 4 — De-rotation stacking panel."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QRadioButton,
    QWidget,
)

from gui.i18n import S
from gui.panels.base_panel import BasePanel
from gui.panels.step_status_widget import StepStatusWidget

_SPINBOX_STYLE = (
    "QDoubleSpinBox { background: #3c3c3c; color: #d4d4d4; border: 1px solid #555;"
    " border-radius: 3px; padding: 3px 6px; }"
    "QDoubleSpinBox:focus { border-color: #4da6ff; }"
)
_CHECK_STYLE = (
    "QCheckBox { color: #d4d4d4; }"
    "QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #666;"
    " border-radius: 2px; background: #3c3c3c; }"
    "QCheckBox::indicator:checked { background: #4da6ff; border-color: #4da6ff; }"
    "QCheckBox::indicator:unchecked { background: #2a2a2a; border-color: #555; }"
)
_RADIO_STYLE = (
    "QRadioButton { color: #d4d4d4; spacing: 5px; }"
    "QRadioButton::indicator { width: 13px; height: 13px; border: 1px solid #666;"
    " border-radius: 7px; background: #2a2a2a; }"
    "QRadioButton::indicator:checked { background: #4da6ff; border-color: #4da6ff; }"
    "QRadioButton:disabled { color: #555; }"
)
_READONLY_STYLE = (
    "QLineEdit { background: #2a2a2a; color: #888; border: 1px solid #3a3a3a;"
    " border-radius: 3px; padding: 3px 6px; }"
)

# ── Panel ──────────────────────────────────────────────────────────────────────

class DerotatePanel(BasePanel):
    STEP_ID   = "04"
    TITLE_KEY = "step04.title"
    DESC_KEY  = "step04.desc"
    OPTIONAL  = False

    def __init__(self, parent: QWidget | None = None) -> None:
        self._output_dir: Path | None = None
        self._input_dir:  Path | None = None
        super().__init__(parent)

    # ── BasePanel interface ───────────────────────────────────────────────────

    def build_form(self) -> None:
        form_widget = QWidget()
        form_widget.setStyleSheet("background: transparent;")
        fl = QFormLayout(form_widget)
        fl.setSpacing(10)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        # Step status dots (auto-derived)
        self._step_status = StepStatusWidget(steps=[3])
        lbl_req = QLabel(S("common.requires"))
        lbl_req.setStyleSheet("color: #888;")
        fl.addRow(lbl_req, self._step_status)

        self._output_lbl = QLineEdit()
        self._output_lbl.setReadOnly(True)
        self._output_lbl.setStyleSheet(_READONLY_STYLE)
        lbl_out = QLabel(S("step04.output_dir"))
        lbl_out.setToolTip(S("step04.output_dir.tooltip"))
        fl.addRow(lbl_out, self._output_lbl)

        # Min quality threshold
        self._min_quality = QDoubleSpinBox()
        self._min_quality.setStyleSheet(_SPINBOX_STYLE)
        self._min_quality.setRange(0.0, 1.0)
        self._min_quality.setDecimals(2)
        self._min_quality.setSingleStep(0.05)
        self._min_quality.setValue(0.05)
        self._min_quality.setToolTip(S("step04.min_quality.tooltip"))
        lbl_mq = QLabel(S("step04.min_quality"))
        lbl_mq.setToolTip(S("step04.min_quality.tooltip"))
        fl.addRow(lbl_mq, self._min_quality)

        # Normalize brightness
        self._normalize = QCheckBox()
        self._normalize.setStyleSheet(_CHECK_STYLE)
        self._normalize.setChecked(False)
        self._normalize.setToolTip(S("step04.normalize.tooltip"))
        lbl_norm = QLabel(S("step04.normalize"))
        lbl_norm.setToolTip(S("step04.normalize.tooltip"))
        fl.addRow(lbl_norm, self._normalize)

        # Satellite composite (multi-rate de-rotation)
        self._satellite_composite = QCheckBox()
        self._satellite_composite.setStyleSheet(_CHECK_STYLE)
        self._satellite_composite.setChecked(False)
        self._satellite_composite.setToolTip(S("step04.satellite_composite.tooltip"))
        lbl_sat = QLabel(S("step04.satellite_composite"))
        lbl_sat.setToolTip(S("step04.satellite_composite.tooltip"))
        from gui.panels.bsp_status import BspStatusRow
        fl.addRow(lbl_sat, BspStatusRow(self._satellite_composite))

        # Mask shape radio buttons (capsule / circular)
        self._mask_capsule  = QRadioButton(S("step04.mask_shape.capsule"))
        self._mask_circular = QRadioButton(S("step04.mask_shape.circular"))
        self._mask_capsule.setStyleSheet(_RADIO_STYLE)
        self._mask_circular.setStyleSheet(_RADIO_STYLE)
        self._mask_capsule.setChecked(True)  # default: capsule
        self._mask_group = QButtonGroup(self)
        self._mask_group.addButton(self._mask_capsule,  0)
        self._mask_group.addButton(self._mask_circular, 1)

        mask_row = QWidget()
        mask_row.setStyleSheet("background: transparent;")
        mask_hl = QHBoxLayout(mask_row)
        mask_hl.setContentsMargins(0, 0, 0, 0)
        mask_hl.setSpacing(14)
        mask_hl.addWidget(self._mask_capsule)
        mask_hl.addWidget(self._mask_circular)
        mask_hl.addStretch()

        lbl_mask = QLabel(S("step04.mask_shape"))
        lbl_mask.setStyleSheet("color: #888;")
        fl.addRow(lbl_mask, mask_row)

        # Enable/disable mask row based on composite checkbox
        def _on_composite_toggled(checked: bool) -> None:
            self._mask_capsule.setEnabled(checked)
            self._mask_circular.setEnabled(checked)
        self._satellite_composite.toggled.connect(_on_composite_toggled)
        _on_composite_toggled(False)  # initially disabled


        idx = self._form_layout.count() - 1
        self._form_layout.insertWidget(idx, form_widget)

    def get_config_updates(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "min_quality_threshold":       self._min_quality.value(),
            "normalize_brightness":        self._normalize.isChecked(),
            "satellite_composite_enabled": self._satellite_composite.isChecked(),
            "composite_mask_shape":        "capsule" if self._mask_capsule.isChecked() else "circular",
        }
        out_text = self._output_lbl.text().strip()
        if out_text:
            result["output_dir"] = str(Path(out_text).parent)
        return result

    def validate(self, config: dict, batch_mode: bool = False) -> list:
        from gui.validation import ValidationIssue, count_files
        from gui.i18n import S
        issues = []
        if not batch_mode:
            input_dir = config.get("input_dir", "").strip()
            if not input_dir or not count_files(input_dir, "*.tif", "*.TIF"):
                issues.append(ValidationIssue("error", S("validate.no_tif_input")))
        rotation_period = float(config.get("rotation_period", 0.0))
        if rotation_period <= 0:
            issues.append(ValidationIssue("error", S("validate.no_rotation_period")))
        horizons_id = str(config.get("horizons_id", "")).strip()
        if not horizons_id:
            issues.append(ValidationIssue("warning", S("validate.no_horizons_id")))
        return issues

    def load_session(self, data: dict[str, Any]) -> None:
        inp = data.get("input_dir", "")
        out = data.get("output_dir", "")
        if inp:
            self._input_dir = Path(inp)
        if out:
            p = Path(out)
            self._output_lbl.setText(str(p / "step04_derotated"))
            self._output_dir = p
            self._step_status.refresh(p)

        self._min_quality.setValue(float(data.get("min_quality_threshold", 0.05)))
        self._normalize.setChecked(bool(data.get("normalize_brightness", False)))
        self._satellite_composite.setChecked(
            bool(data.get("satellite_composite_enabled", False))
        )
        shape = data.get("composite_mask_shape", "capsule")
        self._mask_capsule.setChecked(shape != "circular")
        self._mask_circular.setChecked(shape == "circular")

    def output_paths(self) -> list[Path]:
        if self._output_dir is None:
            return []
        step_dir = self._output_dir / "step04_derotated"
        if not step_dir.exists():
            return []
        paths = sorted(step_dir.rglob("*.tif"))
        return paths[:8]

    def set_output_dir(self, path: Path | str) -> None:
        self._output_dir = Path(path) if path else None
        self._step_status.refresh(self._output_dir)
