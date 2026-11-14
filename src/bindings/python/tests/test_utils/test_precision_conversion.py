# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import openvino as ov
import openvino.opset13 as ops
from openvino import Core, Model, PartialShape, Type
from openvino.utils import disable_conversion, enable_conversion, is_conversion_disabled
from tests.utils.helpers import create_filenames_for_ir


def get_square_model():
    param = ops.parameter(PartialShape([1, 4]), dtype=np.float32, name="data")
    square = ops.multiply(param, param, name="square")
    return Model(square, [param], "square"), square


def test_disable_conversion_to_type():
    _, square = get_square_model()
    assert not is_conversion_disabled(square, Type.f16)

    disable_conversion(square, Type.f16)
    assert is_conversion_disabled(square, Type.f16)
    assert is_conversion_disabled(square, Type.f32, Type.f16)
    assert not is_conversion_disabled(square, Type.bf16)
    assert "DisablePrecisionConversion_0" in square.get_rt_info()

    enable_conversion(square, Type.f16)
    assert not is_conversion_disabled(square, Type.f16)


def test_disable_conversion_from_to_type():
    _, square = get_square_model()

    disable_conversion(square, Type.f32, Type.f16)
    assert is_conversion_disabled(square, Type.f32, Type.f16)
    assert not is_conversion_disabled(square, Type.i32, Type.f16)

    enable_conversion(square, Type.f32, Type.f16)
    assert not is_conversion_disabled(square, Type.f32, Type.f16)


def test_disable_conversion_type_lists():
    _, square = get_square_model()

    disable_conversion(square, [Type.f32, Type.f64], [Type.f16, Type.bf16])
    for from_type in (Type.f32, Type.f64):
        for to_type in (Type.f16, Type.bf16):
            assert is_conversion_disabled(square, from_type, to_type)

    enable_conversion(square, [Type.f32, Type.f64], [Type.f16, Type.bf16])
    for from_type in (Type.f32, Type.f64):
        for to_type in (Type.f16, Type.bf16):
            assert not is_conversion_disabled(square, from_type, to_type)


def test_disable_conversion_survives_serialization(request, tmp_path):
    model, square = get_square_model()
    disable_conversion(square, Type.f16)

    xml_path, _ = create_filenames_for_ir(request.node.name, tmp_path)
    ov.save_model(model, xml_path)
    loaded = Core().read_model(xml_path)

    loaded_square = next(op for op in loaded.get_ops() if op.get_friendly_name() == "square")
    assert is_conversion_disabled(loaded_square, Type.f16)


def test_disable_conversion_keeps_node_precise_in_compiled_model(device):
    # 300 * 300 = 90000 overflows f16 (max 65504); the marked node must stay in f32.
    if device != "GPU":
        pytest.skip("checks the GPU f16 inference path that honors DisablePrecisionConversion")
    core = Core()

    model, square = get_square_model()
    disable_conversion(square, Type.f16)

    compiled = core.compile_model(model, device, {"INFERENCE_PRECISION_HINT": "f16"})
    result = compiled(np.full([1, 4], 300.0, dtype=np.float32))[0]
    assert np.allclose(result, 90000.0)
