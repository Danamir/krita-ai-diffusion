from __future__ import annotations

from PyQt6.QtCore import QMetaObject, Qt, pyqtSignal
from PyQt6.QtGui import QResizeEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..backend.resources import ControlMode
from ..localization import translate as _
from ..model.control import ControlLayer, ControlLayerList, is_krea2_edit
from ..model.control import krea2_grounding_steps, settings_layer_id
from ..model.properties import Binding, bind, bind_combo, bind_toggle
from ..model.root import root
from . import theme
from .interval_slider import IntervalSlider
from .theme import SignalBlocker


class ControlWidget(QWidget):
    def __init__(
        self, control_list: ControlLayerList | None, control: ControlLayer, parent: QWidget
    ):
        super().__init__(parent)
        self._control_list = control_list
        self._control = control
        self._connections: list[QMetaObject.Connection | Binding] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.mode_select = QComboBox(self)
        self.mode_select.setStyleSheet(theme.flat_combo_stylesheet)
        for mode in (m for m in ControlMode if not m.is_internal):
            icon = theme.icon(f"control-{mode.name}")
            self.mode_select.addItem(icon, mode.text, mode)

        self.layer_select = QComboBox(self)
        self.layer_select.setMinimumContentsLength(20)
        self.layer_select.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._update_layers()
        root.active_model.layers.changed.connect(self._update_layers)
        root.active_model.edit_mode_changed.connect(self._update_layers)
        root.active_model.style_changed.connect(self._update_layers)

        self.preset_slider = QSlider(self)
        self.preset_slider.setOrientation(Qt.Orientation.Horizontal)
        self.preset_slider.setMinimumWidth(40)
        self.preset_slider.setRange(0, control.max_preset_value)
        self.preset_slider.setValue(control.preset_value)
        self.preset_slider.setSingleStep(1)
        self.preset_slider.setPageStep(2)
        self.preset_slider.setTickInterval(2)
        self.preset_slider.setTickPosition(QSlider.TickPosition.TicksBothSides)
        self.preset_slider.setToolTip(_("Control strength: how much the layer affects the image"))

        self.error_text = QLabel(self)
        self.error_text.setStyleSheet(f"QLabel {{ color: {theme.yellow}; }}")
        self.error_text.setVisible(not control.is_supported)
        self._set_error(control.error_text)

        self.generate_tool_button = _create_generate_button(
            self, Qt.ToolButtonStyle.ToolButtonIconOnly
        )
        self.generate_tool_button.clicked.connect(control.generate)

        self.generate_regions_tool_button = _create_generate_regions_button(
            self, Qt.ToolButtonStyle.ToolButtonIconOnly
        )
        self.generate_regions_tool_button.clicked.connect(control.generate_segmentation)

        self.add_pose_tool_button = _create_add_pose_button(
            self, Qt.ToolButtonStyle.ToolButtonIconOnly
        )
        self.add_pose_tool_button.clicked.connect(self._add_pose_character)

        self.expand_button = QToolButton(self)
        self.expand_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.expand_button.setIcon(theme.icon("more"))
        self.expand_button.setToolTip(_("Show/hide advanced settings"))
        self.expand_button.setAutoRaise(True)
        self.expand_button.setCheckable(True)
        self.expand_button.setChecked(False)
        self.expand_button.clicked.connect(self._toggle_extended)

        # Krea 2 edit: grounding_px shortcut, takes the place of the preset slider
        self.grounding_slider = QSlider(self)
        self.grounding_slider.setOrientation(Qt.Orientation.Horizontal)
        self.grounding_slider.setMinimumWidth(40)
        self.grounding_slider.setRange(0, len(krea2_grounding_steps) - 1)
        self.grounding_slider.setSingleStep(1)
        self.grounding_slider.setPageStep(1)
        self.grounding_slider.setTickInterval(1)
        self.grounding_slider.setTickPosition(QSlider.TickPosition.TicksBothSides)
        self.grounding_slider.valueChanged.connect(self._set_grounding)

        bar_layout = QHBoxLayout()
        bar_layout.addWidget(self.mode_select)
        bar_layout.addWidget(self.layer_select, 3)
        bar_layout.addWidget(self.generate_tool_button)
        bar_layout.addWidget(self.generate_regions_tool_button)
        bar_layout.addWidget(self.add_pose_tool_button)
        bar_layout.addWidget(self.preset_slider, 1)
        bar_layout.addWidget(self.grounding_slider, 1)
        bar_layout.addWidget(self.error_text, 3)
        bar_layout.addWidget(self.expand_button)
        layout.addLayout(bar_layout)

        if self._control_list is not None:
            self.remove_button = QToolButton(self)
            self.remove_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            self.remove_button.setIcon(theme.icon("remove"))
            self.remove_button.setToolTip(_("Remove control layer"))
            self.remove_button.setAutoRaise(True)
            self.remove_button.clicked.connect(self.remove)
            bar_layout.addWidget(self.remove_button)

        line = QFrame(self)
        line.setObjectName("LeftIndent")
        line.setStyleSheet(f"#LeftIndent {{ color: {theme.line};  }}")
        line.setFrameShape(QFrame.Shape.VLine)
        line.setLineWidth(1)

        extended_layout = QVBoxLayout()
        extended_layout.setContentsMargins(8, 2, 4, 6)
        pad_layout = QHBoxLayout()
        pad_layout.setContentsMargins(10, 0, 0, 0)
        pad_layout.addWidget(line)
        pad_layout.addLayout(extended_layout)

        self.extended_widget = QWidget(self)
        self.extended_widget.setVisible(False)
        self.extended_widget.setLayout(pad_layout)
        layout.addWidget(self.extended_widget)

        self.generate_button = _create_generate_button(
            self.extended_widget, Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.generate_button.clicked.connect(control.generate)

        self.generate_regions_button = _create_generate_regions_button(
            self.extended_widget, Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.generate_regions_button.clicked.connect(control.generate_segmentation)

        self.add_pose_button = _create_add_pose_button(
            self.extended_widget, Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.add_pose_button.clicked.connect(self._add_pose_character)

        self.custom_checkbox = QCheckBox(self.extended_widget)
        self.custom_checkbox.setText(_("Use custom values"))
        self.custom_checkbox.setChecked(control.use_custom_strength)

        actions_layout = QHBoxLayout()
        actions_layout.addWidget(self.custom_checkbox, stretch=1)
        actions_layout.addWidget(self.generate_button)
        actions_layout.addWidget(self.generate_regions_button)
        actions_layout.addWidget(self.add_pose_button)
        extended_layout.addLayout(actions_layout)

        self.strength_slider = QSlider(self.extended_widget)
        self.strength_slider.setOrientation(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 75)
        self.strength_slider.setValue(control.strength)
        self.strength_slider.setSingleStep(1)
        self.strength_slider.setPageStep(10)
        self.strength_label = QLabel("1.0", self.extended_widget)

        self.range_label = QLabel(_("Range") + ":", self.extended_widget)
        self.range_slider = IntervalSlider(
            low=0, high=20, minimum=0, maximum=20, parent=self.extended_widget
        )
        self.range_slider.intervalChanged.connect(self._set_range)
        self.range_start_label = QLabel("0.00", self.extended_widget)
        self.range_end_label = QLabel("1.00", self.extended_widget)

        slider_layout = QGridLayout()
        slider_layout.setSpacing(8)
        self.strength_name_label = QLabel(_("Strength") + ":", self.extended_widget)
        slider_layout.addWidget(self.strength_name_label, 0, 0)
        slider_layout.addWidget(self.strength_slider, 0, 2)
        slider_layout.addWidget(self.strength_label, 0, 3)
        slider_layout.addWidget(self.range_label, 1, 0)
        slider_layout.addWidget(self.range_start_label, 1, 1)
        slider_layout.addWidget(self.range_slider, 1, 2)
        slider_layout.addWidget(self.range_end_label, 1, 3)
        extended_layout.addLayout(slider_layout)

        # Krea 2 edit settings, replace strength and range in Krea 2 edit mode
        self.krea2_widget = QWidget(self.extended_widget)
        krea2_layout = QGridLayout(self.krea2_widget)
        krea2_layout.setContentsMargins(0, 0, 0, 0)
        krea2_layout.setSpacing(8)

        self.krea2_lora_select = QComboBox(self.krea2_widget)
        self.krea2_lora_select.addItem(_("General"), "edit")
        self.krea2_lora_select.addItem(_("Identity"), "identity_edit")
        self.krea2_lora_select.setToolTip(
            _("General: instruction edits (remove, recolor, relight, add, restyle)")
            + "\n"
            + _("Identity: restage a person from a reference")
        )
        krea2_layout.addWidget(QLabel(_("Edit LoRA") + ":", self.krea2_widget), 0, 0)
        krea2_layout.addWidget(self.krea2_lora_select, 0, 1, 1, 2)

        self.ref_boost_slider = BoostSlider(self.krea2_widget)
        self.ref_boost_slider.setToolTip(
            _("Pull harder toward the reference layer (1.0 = off)")
        )
        self.ref_boost_a_slider = BoostSlider(self.krea2_widget)
        self.ref_boost_a_slider.setToolTip(_("Pull harder toward the canvas (1.0 = off)"))
        krea2_layout.addWidget(QLabel(_("Subject boost") + ":", self.krea2_widget), 1, 0)
        krea2_layout.addWidget(self.ref_boost_slider.slider, 1, 1)
        krea2_layout.addWidget(self.ref_boost_slider.spin, 1, 2)
        krea2_layout.addWidget(QLabel(_("Scene boost") + ":", self.krea2_widget), 2, 0)
        krea2_layout.addWidget(self.ref_boost_a_slider.slider, 2, 1)
        krea2_layout.addWidget(self.ref_boost_a_slider.spin, 2, 2)
        krea2_layout.setColumnStretch(1, 1)
        extended_layout.addWidget(self.krea2_widget)

        self._update_grounding()
        self._update_visibility()
        self._update_pose_utils()
        self._update_strength()
        self._update_range()
        self._update_custom_values()

        self._connections = [
            bind_combo(control, "mode", self.mode_select),
            bind_combo(control, "layer_id", self.layer_select),
            bind_toggle(control, "use_custom_strength", self.custom_checkbox),
            bind(control, "preset_value", self.preset_slider, "value"),
            bind(control, "strength", self.strength_slider, "value"),
            control.use_custom_strength_changed.connect(self._update_custom_values),
            control.strength_changed.connect(self._update_strength),
            control.start_changed.connect(self._update_range),
            control.end_changed.connect(self._update_range),
            control.has_active_job_changed.connect(self._update_job_active),
            control.error_text_changed.connect(self._set_error),
            control.is_supported_changed.connect(self._update_visibility),
            control.has_range_changed.connect(self._update_visibility),
            control.can_generate_changed.connect(self._update_visibility),
            control.mode_changed.connect(self._update_visibility),
            control.is_pose_vector_changed.connect(self._update_pose_utils),
            control.layer_id_changed.connect(self._update_visibility),
            control.grounding_px_changed.connect(self._update_grounding),
            bind_combo(control, "krea2_lora", self.krea2_lora_select),
            bind(control, "ref_boost", self.ref_boost_slider.spin, "value"),
            bind(control, "ref_boost_a", self.ref_boost_a_slider.spin, "value"),
            root.active_model.style_changed.connect(self._update_visibility),
            root.active_model.edit_mode_changed.connect(self._update_visibility),
        ]

    def disconnect_all(self):
        Binding.disconnect_all(self._connections)

    def _update_layers(self):
        layers = reversed(root.active_model.layers.images)
        with SignalBlocker(self.layer_select):
            self.layer_select.clear()
            index = -1
            if self._control.is_settings or is_krea2_edit(root.active_model):
                self.layer_select.addItem(_("--Settings--"), settings_layer_id)
                if self._control.is_settings:
                    index = 0
            for layer in layers:
                self.layer_select.addItem(layer.name, layer.id)
                if layer.id == self._control.layer_id:
                    index = self.layer_select.count() - 1
            if index == -1 and self._control_list and self._control in self._control_list:
                self.remove()
            elif index >= 0:
                self.layer_select.setCurrentIndex(index)

    def remove(self):
        assert self._control_list is not None
        self._control_list.remove(self._control)

    def resizeEvent(self, a0: QResizeEvent | None):
        super().resizeEvent(a0)
        self._update_visibility()

    def _add_pose_character(self):
        root.active_model.document.add_pose_character(self._control.layer)

    def _update_visibility(self):
        is_small = self.width() < 420
        is_pose = self._control.mode is ControlMode.pose
        is_segmentation = self._control.mode is ControlMode.segmentation
        is_edit = root.active_model.arch.supports_edit
        is_settings = self._control.is_settings
        is_krea2 = (
            is_krea2_edit(root.active_model)
            and self._control.is_supported
            and (is_settings or self._control.uses_image_reference)
        )
        has_extended = (self._control.is_supported and not is_edit) or is_krea2

        def controls():
            self.mode_select.setEnabled(not is_settings)
            self.layer_select.setVisible(self._control.is_supported or is_settings)
            self.preset_slider.setVisible(self._control.is_supported and not is_edit)
            self.expand_button.setVisible(has_extended)
            self.custom_checkbox.setVisible(not is_krea2)
            self.strength_name_label.setVisible(not is_krea2)
            self.strength_slider.setVisible(not is_krea2)
            self.strength_label.setVisible(not is_krea2)
            self.krea2_widget.setVisible(is_krea2)
            self.grounding_slider.setVisible(is_krea2)
            self.generate_button.setVisible(self._control.can_generate and is_small)
            self.generate_tool_button.setVisible(self._control.can_generate and not is_small)
            self.generate_regions_button.setVisible(
                self._control.is_supported and is_segmentation and is_small
            )
            self.generate_regions_tool_button.setVisible(
                self._control.is_supported and is_segmentation and not is_small
            )
            self.add_pose_button.setVisible(is_pose and is_small)
            self.add_pose_tool_button.setVisible(is_pose and not is_small)
            has_range = self._control.has_range and not is_krea2
            self.range_label.setVisible(has_range)
            self.range_slider.setVisible(has_range)
            self.range_start_label.setVisible(has_range)
            self.range_end_label.setVisible(has_range)
            if not has_extended:
                self.expand_button.setChecked(False)
            self.extended_widget.setVisible(self.expand_button.isChecked())

        def error():
            self.error_text.setVisible(not self._control.is_supported)

        if self._control.is_supported:
            error()
            controls()
        else:  # always hide things to hide first to make space in the layout
            controls()
            error()

    def _update_range(self):
        self.range_slider.setInterval(int(self._control.start * 20), int(self._control.end * 20))
        self.range_start_label.setText(f"{self._control.start:.2f}")
        self.range_end_label.setText(f"{self._control.end:.2f}")

    def _set_range(self, low: int, high: int):
        self._control.start = low / 20
        self._control.end = high / 20

    def _update_strength(self):
        self.strength_label.setText(
            f"{self._control.strength / ControlLayer.strength_multiplier:.2f}"
        )

    def _update_job_active(self):
        self.generate_button.setEnabled(not self._control.has_active_job)
        self.generate_tool_button.setEnabled(not self._control.has_active_job)
        self.generate_regions_button.setEnabled(not self._control.has_active_job)
        self.generate_regions_tool_button.setEnabled(not self._control.has_active_job)
        self.layer_select.setEnabled(not self._control.has_active_job)

    def _update_custom_values(self):
        self.preset_slider.setEnabled(not self._control.use_custom_strength)
        self.strength_slider.setEnabled(self._control.use_custom_strength)
        self.range_slider.setEnabled(self._control.use_custom_strength)

    def _update_pose_utils(self):
        for button in (self.add_pose_button, self.add_pose_tool_button):
            button.setEnabled(self._control.is_pose_vector)
            button.setToolTip(
                _("Add new character pose to selected layer")
                if self._control.is_pose_vector
                else _("Disabled: selected layer must be a vector layer to add a pose")
            )

    def _toggle_extended(self):
        self.extended_widget.setVisible(self.expand_button.isChecked())

    def _set_grounding(self, index: int):
        self._control.grounding_px = krea2_grounding_steps[index]

    def _update_grounding(self):
        px = self._control.grounding_px
        index = min(range(len(krea2_grounding_steps)), key=lambda i: abs(krea2_grounding_steps[i] - px))
        with SignalBlocker(self.grounding_slider):
            self.grounding_slider.setValue(index)
        self.grounding_slider.setToolTip(
            _("Grounding")
            + f": {px} px\n"
            + _("Size of the images seen by the text encoder")
            + "\n"
            + _("Lower: stronger edits, higher: stronger likeness")
        )

    def _set_error(self, error: str):
        parts = error.split("[", 2)
        self.error_text.setText(parts[0])
        if len(parts) > 1:
            self.error_text.setToolTip(
                _("Required model not found, searching for") + f": {parts[1][:-1]}"
            )


def _create_generate_button(parent, style: Qt.ToolButtonStyle):
    button = QToolButton(parent)
    button.setToolButtonStyle(style)
    button.setText(_("From Image"))
    button.setIcon(theme.icon("control-generate"))
    button.setToolTip(_("Generate control layer from current image"))
    return button


def _create_generate_regions_button(parent, style: Qt.ToolButtonStyle):
    button = QToolButton(parent)
    button.setToolButtonStyle(style)
    button.setText(_("From Regions"))
    button.setIcon(theme.icon("region-prompt"))
    button.setToolTip(_("Generate segmentation control layer from current regions"))
    return button


def _create_add_pose_button(parent, style: Qt.ToolButtonStyle):
    button = QToolButton(parent)
    button.setToolButtonStyle(style)
    button.setText(_("Add Skeleton"))
    button.setIcon(theme.icon("add-pose"))
    return button


class BoostSlider:
    """Slider with a spin box for ref_boost values from 0 to 16. The slider is two-part:
    0-2 in 0.05 steps on the first part (1.0 in its middle), 2-16 in 0.5 steps on the rest."""

    fine_steps = 40  # 0.05 steps up to 2.0
    coarse_steps = 28  # 0.5 steps from 2.0 to 16.0

    def __init__(self, parent: QWidget):
        self.slider = QSlider(Qt.Orientation.Horizontal, parent)
        self.slider.setRange(0, self.fine_steps + self.coarse_steps)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(4)
        self.spin = QDoubleSpinBox(parent)
        self.spin.setRange(0.0, 16.0)
        self.spin.setSingleStep(0.05)
        self.spin.setDecimals(2)
        self.spin.setMinimumWidth(60)
        self.slider.valueChanged.connect(self._slider_changed)
        self.spin.valueChanged.connect(self._spin_changed)

    def setToolTip(self, text: str):
        self.slider.setToolTip(text)
        self.spin.setToolTip(text)

    @classmethod
    def to_value(cls, position: int):
        if position <= cls.fine_steps:
            return position * 0.05
        return 2.0 + (position - cls.fine_steps) * 0.5

    @classmethod
    def to_position(cls, value: float):
        if value <= 2.0:
            return round(value / 0.05)
        return cls.fine_steps + round((value - 2.0) / 0.5)

    def _slider_changed(self, position: int):
        if self.to_position(self.spin.value()) != position:
            self.spin.setValue(self.to_value(position))

    def _spin_changed(self, value: float):
        with SignalBlocker(self.slider):
            self.slider.setValue(self.to_position(value))


class ControlListWidget(QWidget):
    _widgets: list[ControlWidget]
    _model: ControlLayerList
    _model_connections: list[QMetaObject.Connection]

    changed = pyqtSignal()

    def __init__(self, model: ControlLayerList, parent=None):
        super().__init__(parent)
        self._model = model
        self._widgets = []
        self._model_connections = []

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: ControlLayerList):
        if self._model != model:
            Binding.disconnect_all(self._model_connections)
            self._model = model
            while len(self._widgets) > 0:
                self._remove_widget(self._widgets[0])
            for control in self._model:
                self._add_widget(control)
            self._model_connections = [
                model.added.connect(self._add_widget),
                model.removed.connect(self._remove_widget),
            ]

    def _add_widget(self, control: ControlLayer):
        widget = ControlWidget(self._model, control, self)
        index = min(control.index, len(self._widgets))
        self._widgets.insert(index, widget)
        self._layout.insertWidget(index, widget)

    def _remove_widget(self, widget: ControlWidget | ControlLayer):
        if isinstance(widget, ControlLayer):
            widget = next(w for w in self._widgets if w._control == widget)
        self._widgets.remove(widget)
        widget.disconnect_all()
        widget.deleteLater()
