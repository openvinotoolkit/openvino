# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from openvino import Core, Model, Type, opset13 as ops


@pytest.mark.parametrize("lower", [-2**63, -2**40, -2**31 - 256, -2**31, -1000, 0])
@pytest.mark.parametrize("cast_input", [False, True])
def test_clamp_i64_lower_bound(device, lower, cast_input):
    # A preceding Convert exercises the GPU reorder/activation fusion where
    # Clamp attributes can remain i64 bounds after the data is lowered to i32.
    # This portable API test supplements the GPU-specific clamp_i64_bounds.cpp.
    dtype = np.float32 if cast_input else np.int64
    values = np.tile(np.array([-1024, -1, 0, 146, 359, 360, 365, 1000], dtype=dtype), 8).reshape(1, 64)
    parameter = ops.parameter(values.shape, dtype=dtype)
    data = ops.convert(parameter, Type.i64) if cast_input else parameter
    clamp = ops.clamp(data, float(lower), 360.0)
    model = Model([clamp], [parameter])
    compiled = Core().compile_model(model, device)
    actual = compiled({0: values})[compiled.output(0)]
    expected = np.maximum(np.minimum(values.astype(np.int64), 360), lower)
    np.testing.assert_array_equal(actual, expected)

    # Check another inference through the same compiled model as well.
    values = np.flip(values, axis=1).copy()
    actual = compiled({0: values})[compiled.output(0)]
    expected = np.maximum(np.minimum(values.astype(np.int64), 360), lower)
    np.testing.assert_array_equal(actual, expected)
