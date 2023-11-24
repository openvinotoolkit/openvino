# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino import Type
import openvino.runtime.opset8 as ov
import numpy as np
import pytest


@pytest.mark.parametrize(("input_shape", "indices", "axis", "expected_shape", "batch_dims"), [
    ((3, 3), (1, 2), [1], [3, 1, 2], []),
    ((3, 3), (1, 2), 1, [3, 1, 2], []),
    ((2, 5), (2, 3), [1], [2, 3], [1]),
    ((2, 5), (2, 3), [1], [2, 2, 3], []),
])
def test_gather(input_shape, indices, axis, expected_shape, batch_dims):
    input_data = ov.parameter(input_shape, name="input_data", dtype=np.float32)
    input_indices = ov.parameter(indices, name="input_indices", dtype=np.int32)
    input_axis = np.array(axis, np.int32)

    node = ov.gather(input_data, input_indices, input_axis, *batch_dims)
    assert node.get_type_name() == "Gather"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape
    assert node.get_output_element_type(0) == Type.f32
