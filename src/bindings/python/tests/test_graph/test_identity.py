# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import openvino.runtime.opset14 as ops
from openvino import PartialShape, Type


@pytest.mark.parametrize(
    ("input_shape", "copy", "expected_output_shape"),
    [
        ([4, 4], False, PartialShape([4, 4])),
        ([10, 8, 8], True, PartialShape([10, 8, 8])),
        ([-1, -1, -1], True, PartialShape([-1, -1, -1])),
        ([10, -1, -1], True, PartialShape([10, -1, -1])),
    ],
)
@pytest.mark.parametrize("op_name", ["identity", "identityOpset15"])
def test_inverse_param_inputs(input_shape, copy, expected_output_shape, op_name):
    data = ops.parameter(input_shape, dtype=np.float32)

    op = ops.inverse(data, copy=copy, name=op_name)
    assert op.get_output_size() == 1
    assert op.get_type_name() == "Identity"
    assert op.get_friendly_name() == op_name
    assert op.get_output_element_type(0) == Type.f32
    assert op.get_output_partial_shape(0) == expected_output_shape

