# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from openvino.opset15 import parameter
from openvino.opset16 import identity
from openvino import PartialShape, Type


@pytest.mark.parametrize(
    ("input_shape", "expected_output_shape"),
    [
        ([4, 4], PartialShape([4, 4])),
        ([10, 8, 8], PartialShape([10, 8, 8])),
        ([-1, -1, -1], PartialShape([-1, -1, -1])),
        ([10, -1, -1], PartialShape([10, -1, -1])),
    ],
)
@pytest.mark.parametrize("op_name", ["identity", "identityOpset16"])
def test_inverse_param_inputs(input_shape, expected_output_shape, op_name):
    data = parameter(input_shape, dtype=np.float32)

    op = identity(data, name=op_name)
    assert op.get_output_size() == 1
    assert op.get_type_name() == "Identity"
    assert op.get_friendly_name() == op_name
    assert op.get_output_element_type(0) == Type.f32
    assert op.get_output_partial_shape(0) == expected_output_shape
