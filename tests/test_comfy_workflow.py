import json
from pathlib import Path

import pytest

from ai_diffusion.backend.comfy_workflow import ComfyObjectInfo, ComfyWorkflow, Output


@pytest.fixture(scope="module")
def object_info() -> dict:
    path = Path(__file__).parent / "data" / "object_info.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture()
def info(object_info: dict) -> ComfyObjectInfo:
    return ComfyObjectInfo(object_info)


def test_contains_and_bool(info: ComfyObjectInfo):
    # sanity: has nodes and truthy
    assert info
    assert "GrowMask" in info
    assert "DoesNotExist" not in info

    # empty info is falsy and contains nothing
    empty = ComfyObjectInfo({})
    assert not empty
    assert ("GrowMask" in empty) is False


def test_inputs_required_and_merged(info: ComfyObjectInfo):
    # Required-only node
    req = info.inputs("GrowMask", "required")
    assert req is not None
    assert set(req.keys()) == {"mask", "expand", "tapered_corners"}

    # Merged (required + optional). The sample has no optional keys, so merged == required
    merged = info.inputs("GrowMask")
    assert merged == req

    # Node with no inputs
    assert info.inputs("ETN_KritaCanvas", "required") is None
    merged_canvas = info.inputs("ETN_KritaCanvas")
    assert isinstance(merged_canvas, dict) and merged_canvas == {}


def test_params_defaults_for_various_types(info: ComfyObjectInfo):
    # GrowMask: INT and BOOLEAN provide defaults
    params = info.params("GrowMask", "required")
    assert params == {"expand": 0, "mask": None, "tapered_corners": True}

    # ETN_KritaOutput: legacy combo (list of values + default dict), picks default
    params_krita_out = info.params("ETN_KritaOutput", "required")
    assert params_krita_out == {"format": "PNG", "images": None}

    # UpscaleModelLoader: COMBO v3 picks first from options
    params_upscale = info.params("UpscaleModelLoader", "required")
    assert params_upscale == {"model_name": "4x_NMKD-Superscale-SP_178000_G.pth"}


def test_options_for_combo_types(info: ComfyObjectInfo):
    # COMBO v3
    opts_model = info.options("UpscaleModelLoader", "model_name")
    assert isinstance(opts_model, list) and len(opts_model) >= 1
    assert opts_model[:3] == [
        "4x_NMKD-Superscale-SP_178000_G.pth",
        "OmniSR_X2_DIV2K.safetensors",
        "OmniSR_X3_DIV2K.safetensors",
    ]

    # Legacy list combo
    opts_format = info.options("ETN_KritaOutput", "format")
    assert opts_format == ["PNG", "JPEG"]

    # Non-existent node/param -> empty list
    assert info.options("DoesNotExist", "foo") == []
    assert info.options("UpscaleModelLoader", "does_not_exist") == []


def _only_node_inputs(w: ComfyWorkflow) -> dict:
    assert len(w.root) == 1
    key = next(iter(w.root))
    return w.root[key]["inputs"]


def test_defaults_integral(info: ComfyObjectInfo):
    w = ComfyWorkflow(node_defs=info)
    # Provide only the non-defaultable required input (mask); omit defaults
    w.add("GrowMask", output_count=1, mask="m")
    inputs = _only_node_inputs(w)
    assert inputs["mask"] == "m"
    assert inputs["expand"] == 0
    assert inputs["tapered_corners"] is True
    assert set(inputs.keys()) == {"mask", "expand", "tapered_corners"}


def test_defaults_combo_v3(info: ComfyObjectInfo):
    w = ComfyWorkflow(node_defs=info)
    # Omit model_name entirely; should use first option from object info
    w.add("UpscaleModelLoader", output_count=1)
    inputs = _only_node_inputs(w)
    assert inputs == {"model_name": "4x_NMKD-Superscale-SP_178000_G.pth"}


def test_defaults_legacy_combo(info: ComfyObjectInfo):
    w = ComfyWorkflow(node_defs=info)
    # Provide required images; omit format to pick default "PNG"
    w.add("ETN_KritaOutput", output_count=1, images="img")
    inputs = _only_node_inputs(w)
    assert inputs["images"] == "img"
    assert inputs["format"] == "PNG"
    assert set(inputs.keys()) == {"images", "format"}


def test_text_encode_qwen_image_21(info: ComfyObjectInfo):
    w = ComfyWorkflow(node_defs=info)
    clip = w.add("None", 1)
    img1 = w.add("None", 1)
    img2 = w.add("None", 1)
    vae = w.add("None", 1)
    cond = w.text_encode_qwen_image_21(clip, "a cat", "bad", [img1, img2], vae)

    assert len(w.root) == 5  # clip + images + vae + encode node
    encode = next(node for node in w if node.type == "TextEncodeQwenImage21")
    assert encode.inputs["clip"] == clip
    assert encode.inputs["prompt"] == "a cat"
    assert encode.inputs["negative_prompt"] == "bad"
    assert encode.inputs["resolution"] == 1024
    assert encode.inputs["vae"] == vae
    assert encode.inputs["images.image_1"] == img1
    assert encode.inputs["images.image_2"] == img2
    assert "images.image_3" not in encode.inputs
    assert "images" not in encode.inputs
    assert cond.positive == Output(encode.id, 0)
    assert cond.negative == Output(encode.id, 1)


def test_text_encode_qwen_image_21_no_vae(info: ComfyObjectInfo):
    w = ComfyWorkflow(node_defs=info)
    clip = w.add("None", 1)
    w.text_encode_qwen_image_21(clip, "a cat", "bad", [])
    encode = next(node for node in w if node.type == "TextEncodeQwenImage21")
    assert "vae" not in encode.inputs
    assert "images.image_1" not in encode.inputs


def test_color_match_strips_alpha(info: ComfyObjectInfo):
    w = ComfyWorkflow(node_defs=info)
    target = w.add("None", 1)
    reference = w.add("None", 1)
    mask = w.add("None", 1)
    out = w.color_match(target, reference, mask, strength=1.0)

    split = [node for node in w if node.type == "SplitImageWithAlpha"]
    assert len(split) == 2
    stripped = {split[0].inputs["image"], split[1].inputs["image"]}
    assert stripped == {target, reference}

    color_match = next(node for node in w if node.type == "INPAINT_ColorMatch")
    stripped_ids = {x[0] for x in [color_match.inputs["target"], color_match.inputs["reference"]]}
    assert stripped_ids == {node.id for node in split}
    assert color_match.inputs["exclude_mask"] == mask
    assert color_match.inputs["strength"] == pytest.approx(1.0)
    assert out.node == color_match.id


def test_color_match_skipped_for_zero_strength(info: ComfyObjectInfo):
    w = ComfyWorkflow(node_defs=info)
    target = w.add("None", 1)
    out = w.color_match(target, w.add("None", 1), strength=0.0)
    assert out == target
    assert not any(node.type in ["SplitImageWithAlpha", "INPAINT_ColorMatch"] for node in w)
