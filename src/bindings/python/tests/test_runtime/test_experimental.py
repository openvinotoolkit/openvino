# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.experimental import evaluate_as_partial_shape, evaluate_both_bounds, set_element_type, set_tensor_type

import pytest
from openvino.runtime import Shape, PartialShape, Dimension, Type
from openvino.runtime.op import Constant
import openvino.runtime.opset13 as ops
import numpy as np


@pytest.fixture
def graph_with_partial_value():
    bounds = [
        (-6, -6), (-5, -4), (-4, 0), (-4, 4), (-4, 3),
        (-3, 4), (0, 0), (0, 2), (1, 1), (1, 42),
    ]
    shape = []
    subtrahend = []

    for (lower, upper) in bounds:
        if lower >= 0:
            shape.append(Dimension(lower, upper))
            subtrahend.append(0)
        else:
            shape.append(Dimension(0, upper - lower))
            subtrahend.append(lower)

    parameter_node = ops.parameter(PartialShape(shape), Type.dynamic)
    shape_of = ops.shape_of(parameter_node)
    subtract = ops.add(
        shape_of,
        Constant(Type.i64, Shape([len(subtrahend)]), subtrahend),
    )
    return subtract


def test_evaluate_both_bounds(graph_with_partial_value):
    node = graph_with_partial_value
    lb, ub = evaluate_both_bounds(node.output(0))
    lower_bounds = [-6, -5, -4, -4, -4, -3, 0, 0, 1, 1]
    upper_bounds = [-6, -4, 0, 4, 3, 4, 0, 2, 1, 42]

    assert np.equal(lb.data, lower_bounds).all()
    assert np.equal(ub.data, upper_bounds).all()


def test_evaluate_as_partial_shape(graph_with_partial_value):
    node = ops.abs(graph_with_partial_value)
    output_value = PartialShape([])
    assert evaluate_as_partial_shape(node.output(0), output_value)


def test_tensor_descriptor():
    data1 = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.int64)
    data2 = np.array([3, 2, 1], dtype=np.int64)

    node = ops.add(data1, data2)
    input_tensor1 = node.get_input_tensor(0)

    assert input_tensor1.get_element_type() == Type.i64
    set_element_type(input_tensor1, Type.f64)
    assert input_tensor1.get_element_type() == Type.f64

    partial_shape = PartialShape([-1, 6])
    set_tensor_type(input_tensor1, Type.i64, partial_shape)
    assert input_tensor1.get_partial_shape() == partial_shape
    assert input_tensor1.get_element_type() == Type.i64
