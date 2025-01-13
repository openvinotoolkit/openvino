# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino.opset1 as ov_opset1
import openvino.opset15 as ov_opset15
import numpy as np
import pytest


def test_squeeze_v1_operator():
    data_shape = [1, 2, 1, 3, 1, 1]
    parameter_data = ov_opset1.parameter(data_shape, name="Data", dtype=np.float32)
    axes = [2, 4]
    model = ov_opset1.squeeze(parameter_data, axes)

    assert model.get_type_name() == "Squeeze"
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [1, 2, 3, 1]


@pytest.mark.parametrize(("input_shape", "axes", "allow_axis_skip", "expected_shape"), [
    ((1, 2, 1, 3, 1, 1), [1, 2, 4], True, [1, 2, 3, 1]),
    ((1, 2, 1, 3, 1, 1), [1, 2, 4], False, [1, 2, 3, 1]),
    ((2, -1, 3), [1], False, [2, 3])
])
def test_squeeze_v15_operator(input_shape, axes, allow_axis_skip, expected_shape):
    parameter_data = ov_opset15.parameter(input_shape, name="Data", dtype=np.float32)
    model = ov_opset15.squeeze(parameter_data, axes, allow_axis_skip, name="Squeeze")

    assert model.get_type_name() == "Squeeze"
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == expected_shape


def test_squeeze_v15_dynamic_rank_output():
    parameter_data = ov_opset15.parameter((2, -1, 3), name="Data", dtype=np.float32)
    model = ov_opset15.squeeze(parameter_data, [1], True, name="Squeeze")

    assert model.get_type_name() == "Squeeze"
    assert model.get_output_size() == 1
    assert model.get_output_partial_shape(0).to_string() == "[...]"


def test_squeeze_v15_axes_not_given():
    parameter_data = ov_opset15.parameter((1, 3, 1, 1, 3, 5), name="Data", dtype=np.float32)
    model = ov_opset15.squeeze(data=parameter_data, name="Squeeze")

    assert model.get_type_name() == "Squeeze"
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [3, 3, 5]
