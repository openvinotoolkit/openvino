# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.dev_api import evaluate_as_partial_shape, evaluate_both_bounds

from openvino.runtime import Shape, PartialShape, Dimension, Type
from openvino.runtime.op import Constant
import openvino.runtime.opset13 as ops


def construct_graph_with_partial_value():
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


def test_evaluate_as_partial_shape():
    node = ops.abs(construct_graph_with_partial_value())
    output_value = PartialShape([])
    assert evaluate_as_partial_shape(node.output(0), output_value)
