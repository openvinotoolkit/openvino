# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from tests.runtime import get_runtime
from tests.tests_python.utils import import_onnx_model, run_model


def make_quantize_linear_model(data, scale, zero_point, runtime_zero_point=False, **attributes):
    output_type = helper.np_dtype_to_tensor_dtype(zero_point.dtype)
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, data.shape)]
    initializers = [numpy_helper.from_array(scale, name="scale")]
    if runtime_zero_point:
        inputs.append(helper.make_tensor_value_info("zero_point", output_type, zero_point.shape))
    else:
        initializers.append(numpy_helper.from_array(zero_point, name="zero_point"))

    node = helper.make_node("QuantizeLinear", ["x", "scale", "zero_point"], ["y"], **attributes)
    graph = helper.make_graph(
        [node],
        "quantize_linear_rounding",
        inputs,
        [helper.make_tensor_value_info("y", output_type, data.shape)],
        initializer=initializers,
    )
    return helper.make_model(
        graph,
        producer_name="OpenVINO ONNX Frontend",
        opset_imports=[helper.make_opsetid("", 13)],
    )


@pytest.mark.parametrize("dtype", [np.uint8, np.int8])
def test_quantize_linear_constant_odd_zero_point(dtype):
    limits = np.iinfo(dtype)
    signed = limits.min < 0
    zero_point = np.array(-1 if signed else 1, dtype=dtype)
    scale = np.array(0.5, dtype=np.float32)
    data = np.array([-100000, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 100000], dtype=np.float32)
    expected = np.array(
        [limits.min, -3, -3, -1, -1, 1, 1, limits.max]
        if signed else [limits.min, 0, 0, 1, 1, 3, 3, limits.max],
        dtype=dtype,
    )

    model = make_quantize_linear_model(data, scale, zero_point)
    result = run_model(model, [data])[0]

    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result, expected)


def test_quantize_linear_runtime_zero_point_parity():
    zero_points = [128, 129, 130, 1]
    scale = np.array(0.5, dtype=np.float32)
    data = np.array([-1000, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1000], dtype=np.float32)
    model = make_quantize_linear_model(
        data, scale, np.array(zero_points[0], dtype=np.uint8), runtime_zero_point=True,
    )
    computation = get_runtime().computation(import_onnx_model(model))

    for value in zero_points:
        zero_point = np.array(value, dtype=np.uint8)
        expected = np.clip(np.rint(data / scale) + value, 0, 255).astype(np.uint8)
        result = computation(data, zero_point)[0]

        assert result.dtype == expected.dtype
        np.testing.assert_array_equal(result, expected)


def test_quantize_linear_per_axis_mixed_zero_point_parity():
    zero_point = np.array([1, 2, 3], dtype=np.uint8)
    scale = np.array([0.5, 1, 2], dtype=np.float32)
    broadcast_scale = scale.reshape(1, 3, 1)
    broadcast_zero_point = zero_point.astype(np.float32).reshape(1, 3, 1)
    quotients = np.array([-1000, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 1000], dtype=np.float32)
    data = np.broadcast_to(quotients * broadcast_scale, (2, 3, 8)).copy()
    expected = np.clip(
        np.rint(data / broadcast_scale) + broadcast_zero_point, 0, 255,
    ).astype(np.uint8)

    model = make_quantize_linear_model(data, scale, zero_point, axis=1)
    result = run_model(model, [data])[0]

    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result, expected)
