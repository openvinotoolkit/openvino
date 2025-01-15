# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import openvino.opset13 as ov
from openvino import Type


def test_scatter_update_props():
    dtype = np.int8
    parameter_r = ov.parameter([2, 3, 4], dtype=dtype, name="data")
    parameter_i = ov.parameter([2, 1], dtype=dtype, name="indices")
    parameter_u = ov.parameter([2, 2, 1, 4], dtype=dtype, name="updates")
    axis = np.array([1], dtype=np.int8)

    node = ov.scatter_update(parameter_r, parameter_i, parameter_u, axis)
    assert node.get_type_name() == "ScatterUpdate"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [2, 3, 4]
    assert node.get_output_element_type(0) == Type.i8


def test_scatter_update_elements_default_values():
    dtype = np.int8
    parameter_r = ov.parameter([2, 4, 5, 7], dtype=dtype, name="data")
    parameter_i = ov.parameter([2, 2, 2, 2], dtype=dtype, name="indices")
    parameter_u = ov.parameter([2, 2, 2, 2], dtype=dtype, name="updates")
    axis = np.array([1], dtype=np.int8)

    node = ov.scatter_elements_update(parameter_r, parameter_i, parameter_u, axis)
    assert node.get_type_name() == "ScatterElementsUpdate"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [2, 4, 5, 7]
    assert node.get_output_element_type(0) == Type.i8


@pytest.mark.parametrize(
    "reduction",
    [
        "none",
        "sum",
        "prod",
        "min",
        "max",
        "mean",
    ],
)
def test_scatter_update_elements_props(reduction):
    dtype = np.int8
    parameter_r = ov.parameter([2, 4, 5, 7], dtype=dtype, name="data")
    parameter_i = ov.parameter([2, 2, 2, 2], dtype=dtype, name="indices")
    parameter_u = ov.parameter([2, 2, 2, 2], dtype=dtype, name="updates")
    axis = np.array([1], dtype=np.int8)

    node = ov.scatter_elements_update(parameter_r, parameter_i, parameter_u, axis, reduction, False)
    assert node.get_type_name() == "ScatterElementsUpdate"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [2, 4, 5, 7]
    assert node.get_output_element_type(0) == Type.i8
