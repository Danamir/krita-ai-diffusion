from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from PyQt6.QtCore import QObject, Qt, QUuid, pyqtSignal
from PyQt6.QtGui import QColor

from .. import util
from ..backend import resources
from ..backend.api import ControlInput, Krea2EditInput
from ..backend.resources import Arch, ControlMode, ResourceKind, resource_id
from ..image import BlendMode, Bounds, Extent, Image
from ..layer import Layer, LayerType
from ..localization import translate as _
from ..util import PluginError
from ..util import client_logger as log
from . import jobs
from .properties import ObservableProperties, Property

if TYPE_CHECKING:
    from .model import DocumentModel

# Virtual layer of the "--Settings--" control layer: holds Krea 2 edit settings without an image
settings_layer_id = QUuid("{4b2f7c1e-8d3a-4e6b-9f05-2c7a1e9d6b38}")
krea2_grounding_steps = (512, 768, 1024, 1536)


def is_krea2_edit(model: DocumentModel):
    return model.arch is Arch.krea2 and model.is_editing


class ControlLayer(QObject, ObservableProperties):
    max_preset_value = 4
    strength_multiplier = 50
    clip_vision_extent = Extent(224, 224)

    mode = Property(ControlMode.reference, persist=True, setter="set_mode")
    layer_id = Property(QUuid(), persist=True)
    preset_value = Property(2, persist=True, setter="set_preset_value")
    strength = Property(50, persist=True)
    start = Property(0.0, persist=True)
    end = Property(1.0, persist=True)
    use_custom_strength = Property(False, persist=True, setter="set_use_custom_strength")
    is_supported = Property(True)
    is_pose_vector = Property(False)
    can_generate = Property(True)
    has_range = Property(True)
    has_active_job = Property(False)
    error_text = Property("")
    krea2_lora = Property("edit", persist=True)
    grounding_px = Property(768, persist=True)
    ref_boost = Property(1.0, persist=True)
    ref_boost_a = Property(1.0, persist=True)

    mode_changed = pyqtSignal(ControlMode)
    layer_id_changed = pyqtSignal(QUuid)
    preset_value_changed = pyqtSignal(int)
    strength_changed = pyqtSignal(int)
    start_changed = pyqtSignal(float)
    end_changed = pyqtSignal(float)
    use_custom_strength_changed = pyqtSignal(bool)
    is_supported_changed = pyqtSignal(bool)
    is_pose_vector_changed = pyqtSignal(bool)
    can_generate_changed = pyqtSignal(bool)
    has_active_job_changed = pyqtSignal(bool)
    has_range_changed = pyqtSignal(bool)
    error_text_changed = pyqtSignal(str)
    krea2_lora_changed = pyqtSignal(str)
    grounding_px_changed = pyqtSignal(int)
    ref_boost_changed = pyqtSignal(float)
    ref_boost_a_changed = pyqtSignal(float)
    modified = pyqtSignal(QObject, str)

    def __init__(self, model: DocumentModel, mode: ControlMode, layer_id: QUuid, index: int):
        from .root import root

        super().__init__()
        self._model = model
        self._index = index
        self._list: ControlLayerList | None = None
        self._generate_job: jobs.Job | None = None
        self.layer_id = layer_id
        self.mode = mode
        self._update_is_supported()

        self.mode_changed.connect(self._update_is_supported)
        model.style_changed.connect(self._update_is_supported)
        model.edit_mode_changed.connect(self._update_is_supported)
        root.connection.state_changed.connect(self._update_is_supported)
        self.layer_id_changed.connect(self._update_is_pose_vector)
        model.jobs.job_finished.connect(self._update_active_job)

    @property
    def layer(self):
        layer = self._model.layers.updated().find(self.layer_id)
        assert layer is not None, "Control layer has been deleted"
        return layer

    @property
    def is_settings(self):
        return self.layer_id == settings_layer_id

    @property
    def uses_image_reference(self):
        """True if the layer's image is passed as a reference image to edit models."""
        if self.is_settings:
            return False
        return self.mode.is_ip_adapter or self.mode.can_substitute_instruction(self._model.arch)

    def to_krea2_edit(self):
        return Krea2EditInput(self.krea2_lora, self.grounding_px, self.ref_boost, self.ref_boost_a)

    def set_mode(self, mode: ControlMode):
        if mode != self.mode:
            self._mode = mode
            self.mode_changed.emit(mode)
            self._update_is_pose_vector()
            if not self.use_custom_strength:
                self._set_values_from_preset()

    def set_preset_value(self, value: int):
        if value != self.preset_value:
            self._preset_value = value
            self.preset_value_changed.emit(value)
            self._set_values_from_preset()

    def _set_values_from_preset(self):
        params = ControlPresets.instance().interpolate(
            self.mode, self._model.arch, self.preset_value / self.max_preset_value
        )
        self.strength = int(params.strength * self.strength_multiplier)
        self.start, self.end = params.range

    def set_use_custom_strength(self, value: bool):
        if value != self.use_custom_strength:
            self._use_custom_strength = value
            self.use_custom_strength_changed.emit(value)
            if not value:
                self._set_values_from_preset()

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index: int):
        self._index = index
        self._update_is_supported()

    def to_api(self, bounds: Bounds | None = None, time: int | None = None):
        assert not self.is_settings, "Settings control layer has no image"
        layer = self.layer
        if not self.is_supported:
            raise PluginError(f"Can't use '{layer.name}' as control layer: {self.error_text}")

        extent = bounds.extent if bounds else self._model.document.extent
        if self.mode.is_ip_adapter and not layer.bounds.is_zero:
            bounds = None  # ignore mask bounds, use layer bounds

        image = layer.get_pixels(bounds, time)

        if self.mode is ControlMode.segmentation:
            image.make_opaque(background=Qt.GlobalColor.white)

        if self.mode.is_lines or self.mode is ControlMode.stencil:
            image.make_opaque(background=Qt.GlobalColor.white)

        if self.mode.is_ip_adapter:
            if self._model.arch.supports_edit:
                if image.extent.height > extent.height:
                    w = (image.extent.width * extent.height) // image.extent.height
                    image = Image.scale(image, Extent(w, extent.height))
            else:
                image = Image.scale(image, self.clip_vision_extent)

        strength = self.strength / self.strength_multiplier
        return ControlInput(self.mode, image, strength, (self.start, self.end))

    def generate_segmentation(self):
        assert self.mode is ControlMode.segmentation

        ok, msg = self._model.document.check_color_mode()
        if not ok and msg:
            self._model.report_error(msg)
            return

        try:
            bounds = Bounds.from_extent(self._model.document.extent)
            image = self._segmentation_image_from_regions(bounds)
            if image is None:
                self._model.report_error(_("Text prompt regions have not been set up."))
                return

            layer = self._model.layers.create(f"[Control] {self.mode.text}", image, bounds)
            self.layer_id = layer.id
        except Exception as e:
            self._model.report_error(util.log_error(e))
        else:
            self._model.clear_error()

    def _segmentation_image_from_regions(self, bounds: Bounds):
        from .region import RegionLink

        image = Image.create(bounds.extent, fill=Qt.GlobalColor.white)
        has_region_layer = False
        root = self._model.active_regions

        layers = [
            layer
            for layer in root.layers.all
            if root.find_linked(layer, RegionLink.direct) is not None
            and Bounds.intersection(bounds, layer.compute_bounds()).area > 0
        ]

        for index, layer in enumerate(layers):
            color = _segmentation_colors[index % len(_segmentation_colors)]
            region_image = self._segmentation_region_image(layer, bounds, color)
            image.draw_image(region_image, blend=BlendMode.alpha)
            has_region_layer = True

        return image if has_region_layer else None

    def _segmentation_region_image(self, layer: Layer, bounds: Bounds, color: tuple[int, int, int]):
        mask = layer.get_mask(bounds)
        image = Image.create(bounds.extent, fill=QColor(*color, 255))
        image._qimage.setAlphaChannel(mask._qimage)
        return image

    def generate(self):
        self._generate_job = self._model.generate_control_layer(self)
        self.has_active_job = True

    def _update_is_supported(self):
        from .root import root

        is_supported = True
        if self.is_settings:
            if not is_krea2_edit(self._model):
                self.error_text = _("Only used in Krea 2 edit mode")
                is_supported = False
            self.is_supported = is_supported
            self.can_generate = False
            return

        if client := root.connection.client_if_connected:
            models = client.models.for_arch(self._model.arch)

            if self.mode.is_ip_adapter and models.arch in [Arch.illu, Arch.illu_v]:
                resid = resource_id(ResourceKind.clip_vision, Arch.illu, "ip_adapter")
                has_clip_vision = client.models.resources.get(resid, None) is not None
                if not has_clip_vision:
                    search = resources.search_path(
                        ResourceKind.clip_vision, Arch.illu, "ip_adapter"
                    )
                    self.error_text = _("The server is missing the ClipVision model") + f" {search}"
                    is_supported = False

            if self.mode.is_ip_adapter and models.arch.supports_edit:
                is_supported = True  # Reference images are merged into the conditioning context
            elif self.mode.is_ip_adapter and models.ip_adapter.find(self.mode) is None:
                search_path = resources.search_path(ResourceKind.ip_adapter, models.arch, self.mode)
                if search_path:
                    self.error_text = (
                        _("The server is missing the IP-Adapter model") + f" {self.mode.text}"
                    )
                else:
                    self.error_text = _("Not supported for") + f" {models.arch.value}"
                if not client.features.ip_adapter:
                    self.error_text = _("IP-Adapter is not supported by this GPU")
                is_supported = False
            elif self.mode.is_control_net and models.arch.supports_edit:
                is_supported = self.mode.can_substitute_instruction(models.arch)
                if not is_supported:
                    self.error_text = _("Not supported for") + f" {models.arch.value}"
            elif self.mode.is_control_net:
                model = models.find_control(self.mode)
                self.has_range = model == models.control.find(self.mode, True)
                if model is None:
                    search_arch = Arch.illu if models.arch is Arch.illu_v else models.arch
                    search_path = (
                        resources.search_path(ResourceKind.controlnet, search_arch, self.mode)
                        or resources.search_path(ResourceKind.model_patch, search_arch, self.mode)
                        or resources.search_path(ResourceKind.lora, models.arch, self.mode)
                    )
                    if search_path:
                        self.error_text = (
                            _("The ControlNet model is not installed") + f" {search_path}"
                        )
                    else:
                        self.error_text = _("Not supported for") + f" {models.arch.value}"
                    is_supported = False

            if is_supported and models.arch is Arch.krea2 and self.uses_image_reference:
                if not self._model.is_editing:
                    self.error_text = _("Krea 2 supports reference layers in edit mode only")
                    is_supported = False
                elif self._has_previous_image_reference():
                    self.error_text = _("Krea 2 edit supports only one reference layer")
                    is_supported = False

            if self._index >= client.features.max_control_layers:
                self.error_text = _("Too many control layers")
                is_supported = False

        self.is_supported = is_supported
        self.can_generate = is_supported and self.mode.has_preprocessor

    def _has_previous_image_reference(self):
        if self._list is None:
            return False
        previous = (c for c in self._list if c is not self and c.index < self.index)
        return any(c.uses_image_reference for c in previous)

    def _update_is_pose_vector(self):
        if self.is_settings:
            self.is_pose_vector = False
            return
        self.is_pose_vector = self.mode is ControlMode.pose and self.layer.type is LayerType.vector

    def _update_active_job(self):
        from .jobs import JobState

        active = not (self._generate_job is None or self._generate_job.state is JobState.finished)
        if self.has_active_job and not active:
            self._job = None  # job done
        self.has_active_job = active


class ControlLayerList(QObject):
    """List of control layers for one document."""

    added = pyqtSignal(ControlLayer)
    removed = pyqtSignal(ControlLayer)

    def __init__(self, model: DocumentModel):
        super().__init__()
        self._model = model
        self._layers: list[ControlLayer] = []
        self._last_mode = ControlMode.scribble
        self._model.layers.removed.connect(self._remove_layer)

    def add(self):
        layer = self._model.layers.active
        if layer.type.is_filter and layer.parent_layer and not layer.parent_layer.is_root:
            layer = layer.parent_layer
        if not layer.type.is_image:
            layer = next(iter(self._model.layers.images), None)
        if layer is None:  # shouldn't be possible, Krita doesn't allow removing all non-mask layers
            log.warning("Trying to add control layer, but document has no suitable layer")
            return
        is_edit = self._model.arch.is_edit or is_krea2_edit(self._model)
        mode = ControlMode.reference if is_edit else self._last_mode
        control = ControlLayer(self._model, mode, layer.id, len(self._layers))
        self._insert(control, len(self._layers))

    def add_settings(self):
        """Adds the virtual "--Settings--" control layer on top of the list."""
        control = ControlLayer(self._model, ControlMode.reference, settings_layer_id, 0)
        self._insert(control, 0)
        return control

    def _insert(self, control: ControlLayer, index: int):
        control._list = self
        control.mode_changed.connect(self._update_last_mode)
        control.mode_changed.connect(self._update_supported)
        control.layer_id_changed.connect(self._update_supported)
        self._layers.insert(index, control)
        for i, c in enumerate(self._layers):
            c.index = i
        self.added.emit(control)

    def _update_supported(self):
        for c in self._layers:
            c._update_is_supported()

    @property
    def settings(self):
        return next((c for c in self._layers if c.is_settings), None)

    def krea2_edit_input(self):
        """Krea 2 edit settings from the topmost settings or reference control layer."""
        if not is_krea2_edit(self._model):
            return None
        for c in self._layers:
            if c.is_settings or (c.uses_image_reference and c.is_supported):
                return c.to_krea2_edit()
        return None

    def emplace(self):
        self.add()
        return self[-1]

    def remove(self, control: ControlLayer):
        self._layers.remove(control)
        self.removed.emit(control)

        for i, c in enumerate(self._layers):
            c.index = i

    def to_api(self, bounds: Bounds | None = None, time: int | None = None):
        layers = [c for c in self._layers if not c.is_settings]
        for layer in (c for c in layers if not c.is_supported):
            log.warning(f"Trying to use control layer {layer.mode.name}: {layer.error_text}")
        return [c.to_api(bounds, time) for c in layers if c.is_supported]

    def _update_last_mode(self, mode: ControlMode):
        self._last_mode = mode

    def _remove_layer(self, layer: Layer):
        if control := next((c for c in self._layers if c.layer_id == layer.id), None):
            self.remove(control)

    def __len__(self):
        return len(self._layers)

    def __getitem__(self, i):
        return self._layers[i]

    def __iter__(self):
        return iter(self._layers)


class ControlParams(NamedTuple):
    strength: float
    range: tuple[float, float]

    @staticmethod
    def from_dict(data: dict[str, Any]):
        return ControlParams(data["strength"], (data["start"], data["end"]))


class ControlPresets:
    _path: Path
    _user_path: Path
    _presets: dict[str, dict[str, list[dict[str, Any]]]]

    _instance: ControlPresets | None = None

    @classmethod
    def instance(cls) -> ControlPresets:
        if cls._instance is None:
            cls._instance = ControlPresets()
        return cls._instance

    def __init__(self):
        self._path = util.plugin_dir / "presets" / "control.json"
        self._user_path = util.user_data_dir / "presets" / "control.json"
        self._read()

    def get(self, mode: ControlMode, arch: Arch):
        default = self._presets["default"]
        versions = self._presets.get(mode.name, default)
        all = versions.get("all", None)
        presets = versions.get(arch.name, all)
        if presets is None:
            raise PluginError(f"No control strength presets found for {mode} and {arch}")
        return [ControlParams.from_dict(p) for p in presets]

    def interpolate(self, mode: ControlMode, arch: Arch, value: float):
        assert value >= 0 and value <= 1, f"Interpolate value out of range: {value}"
        presets = self.get(mode, arch)
        if len(presets) == 1 or value <= 0:
            return presets[0]
        if value == 1:
            return presets[-1]
        value = value * (len(presets) - 1)
        for i, p0 in enumerate(presets):
            if value < i + 1:
                p1 = presets[i + 1]
                t = value - i
                return ControlParams(
                    _lerp(p0.strength, p1.strength, t),
                    (_lerp(p0.range[0], p1.range[0], t), _lerp(p0.range[1], p1.range[1], t)),
                )
        assert False, f"Interpolation failed: {mode}, {arch}, value={value}, presets={presets}"

    def _read(self):
        self._presets = self._read_file(self._path)
        _validate_presets(self._path, self._presets)
        if self._user_path.exists():
            user = self._read_file(self._user_path)
            if _validate_presets(self._user_path, user):
                _recursive_update(self._presets, user)
        else:
            self._user_path.parent.mkdir(parents=True, exist_ok=True)
            self._user_path.write_text(json.dumps({}, indent=4))

    def _read_file(self, path: Path):
        try:
            return json.load(path.open("r"))
        except Exception as e:
            raise ValueError(f"Failed to read control layer presets file {path}: {e}") from e


def _validate_presets(filepath: Path, data: dict[str, Any]) -> bool:
    control_modes = ["default"] + list(ControlMode.__members__.keys())
    model_archs = list(Arch.__members__.keys())

    for mode, versions in data.items():
        if mode not in control_modes:
            log.error(
                f"Invalid control mode '{mode}' in presets file {filepath}."
                f" Valid modes are: {', '.join(control_modes)}"
            )
            return False
        if not isinstance(versions, dict):
            log.error(f"Invalid presets for mode '{mode}' in presets file {filepath}.")
            return False
        for arch, presets in versions.items():
            if arch not in model_archs:
                log.error(
                    f"Invalid Base model '{arch}' for mode '{mode}' in presets file {filepath}."
                    f" Valid versions are: {', '.join(model_archs)}"
                )
                return False
            if not isinstance(presets, list):
                log.error(
                    f"Invalid presets for '{mode}/{arch}' in presets file {filepath}."
                    f" Expected a list, got {presets}"
                )
                return False
            for p in presets:
                if not isinstance(p, dict) or not all(k in p for k in ("strength", "start", "end")):
                    log.error(
                        f"Invalid preset for '{mode}/{arch}' in presets file {filepath}."
                        f" Expected a {{strength, start, end}}, got {p}"
                    )
                    return False
    return True


control_mode_text = {
    ControlMode.reference: _("Reference"),
    ControlMode.inpaint: _("Inpaint"),
    ControlMode.style: _("Style"),
    ControlMode.composition: _("Composition"),
    ControlMode.face: _("Face"),
    ControlMode.universal: _("Universal"),
    ControlMode.scribble: _("Scribble"),
    ControlMode.line_art: _("Line Art"),
    ControlMode.soft_edge: _("Soft Edge"),
    ControlMode.canny_edge: _("Canny Edge"),
    ControlMode.depth: _("Depth"),
    ControlMode.normal: _("Normal"),
    ControlMode.pose: _("Pose"),
    ControlMode.segmentation: _("Segment"),
    ControlMode.blur: _("Unblur"),
    ControlMode.stencil: _("Stencil"),
    ControlMode.hands: _("Hands"),
}

_segmentation_colors = [
    (255, 255, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (80, 80, 80),
    (160, 80, 0),
    (80, 160, 0),
    (0, 80, 160),
    (0, 160, 80),
]


def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def _recursive_update(a: dict[str, Any], b: dict[str, Any]):
    for k, v in b.items():
        if isinstance(v, dict):
            a[k] = _recursive_update(a.get(k, {}), v)
        else:
            a[k] = v
    return a
