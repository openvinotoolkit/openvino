# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import re

import openvino.runtime.opset13 as ov
from openvino import Type


@pytest.mark.parametrize("op_name", [
    "ABC",
    "Fakeee",
    "123456",
    "FakeQuantize",
])
def test_affix_not_applied_on_nodes(op_name):
    levels = np.int32(4)
    data_shape = [1, 2, 3, 4]
    bound_shape = []

    data_name = "data"
    parameter_data = ov.parameter(data_shape, name=data_name, dtype=np.float32)

    input_low_name = "input_low"
    parameter_input_low = ov.parameter(bound_shape, name=input_low_name, dtype=np.float32)

    input_high_name = "input_high"
    parameter_input_high = ov.parameter(bound_shape, name=input_high_name, dtype=np.float32)

    output_low_name = "output_low"
    parameter_output_low = ov.parameter(bound_shape, name=output_low_name, dtype=np.float32)

    output_high_name = "output_high"
    parameter_output_high = ov.parameter(bound_shape, name=output_high_name, dtype=np.float32)

    model = ov.fake_quantize(
        parameter_data,
        parameter_input_low,
        parameter_input_high,
        parameter_output_low,
        parameter_output_high,
        levels,
        name=op_name,
    )

    # Check if node was created correctly
    assert model.get_type_name() == "FakeQuantize"
    assert model.get_friendly_name() == op_name
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [1, 2, 3, 4]

    assert data_name == parameter_data.friendly_name
    assert input_low_name == parameter_input_low.friendly_name
    assert input_high_name == parameter_input_high.friendly_name
    assert output_low_name == parameter_output_low.friendly_name
    assert output_high_name == parameter_output_high.friendly_name


@pytest.mark.parametrize("op_name", [
    "ABC",
    "Fakeee",
    "123456",
    "FakeQuantize",
])
def test_affix_not_applied_on_output(op_name):
    levels = np.int32(4)
    data_shape = [1, 2, 3, 4]
    bound_shape = []

    p1 = ov.parameter(data_shape, name="A", dtype=np.float32)
    p2 = ov.parameter(data_shape, name="B", dtype=np.float32)

    data_name = "AplusB"
    data = ov.add(p1, p2, name=data_name)
    data_output = data.output(0)

    input_low_name = "input_low"
    parameter_input_low = ov.parameter(bound_shape, name=input_low_name, dtype=np.float32)

    input_high_name = "input_high"
    parameter_input_high = ov.parameter(bound_shape, name=input_high_name, dtype=np.float32)

    output_low_name = "output_low"
    parameter_output_low = ov.parameter(bound_shape, name=output_low_name, dtype=np.float32)

    output_high_name = "output_high"
    parameter_output_high = ov.parameter(bound_shape, name=output_high_name, dtype=np.float32)

    model = ov.fake_quantize(
        data_output,
        parameter_input_low,
        parameter_input_high,
        parameter_output_low,
        parameter_output_high,
        levels,
        name=op_name,
    )

    # Check if node was created correctly
    assert model.get_type_name() == "FakeQuantize"
    assert model.get_friendly_name() == op_name
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [1, 2, 3, 4]

    assert data_name == data_output.get_node().friendly_name
    assert input_low_name == parameter_input_low.friendly_name
    assert input_high_name == parameter_input_high.friendly_name
    assert output_low_name == parameter_output_low.friendly_name
    assert output_high_name == parameter_output_high.friendly_name


@pytest.mark.parametrize("op_name", [
    "ABC",
    "Fakeee",
    "123456",
    "FakeQuantize",
])
def test_fake_quantize_prefix(op_name):
    levels = np.int32(4)
    data_shape = [1, 2, 3, 4]
    bound_shape = [1]

    a_arr = np.ones(data_shape, dtype=np.float32)
    b_arr = np.array(bound_shape, dtype=np.float32)
    c_arr = np.array(bound_shape, dtype=np.float32)
    d_arr = np.array(bound_shape, dtype=np.float32)
    e_arr = np.array(bound_shape, dtype=np.float32)

    model = ov.fake_quantize(
        a_arr,
        b_arr,
        c_arr,
        d_arr,
        e_arr,
        levels,
        name=op_name,
    )

    # Check if node was created correctly
    assert model.get_type_name() == "FakeQuantize"
    assert model.get_friendly_name() == op_name
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [1, 2, 3, 4]
    # Check that other parameters change:
    for node_input in model.inputs():
        generated_node = node_input.get_source_output().get_node()
        assert op_name + "/" in generated_node.friendly_name
