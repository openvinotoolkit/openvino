# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx
import pytest
from onnx.helper import make_graph, make_model, make_tensor_value_info

from tests.tests_python.utils import run_model


def import_and_compute(op_type, input_data_left, input_data_right, opset=7, **node_attributes):
    inputs = [np.array(input_data_left), np.array(input_data_right)]
    onnx_node = onnx.helper.make_node(op_type, inputs=["x", "y"], outputs=["z"], **node_attributes)

    input_tensors = [
        make_tensor_value_info(name, onnx.TensorProto.FLOAT, value.shape)
        for name, value in zip(onnx_node.input, inputs)
    ]
    output_tensors = [make_tensor_value_info(name, onnx.TensorProto.FLOAT, ()) for name in onnx_node.output]

    graph = make_graph([onnx_node], "compute_graph", input_tensors, output_tensors)
    model = make_model(graph, producer_name="OpenVINO ONNX Frontend")
    model.opset_import[0].version = opset
    inputs = [i.astype(np.float32) for i in inputs]  # WA for new Python API
    return run_model(model, inputs)[0]


def test_add_opset4():
    assert np.array_equal(import_and_compute("Add", 1, 2, opset=4), np.array(3, dtype=np.float32))

    assert np.array_equal(import_and_compute("Add", [1], [2], opset=4), np.array([3], dtype=np.float32))

    assert np.array_equal(import_and_compute("Add", [1, 2], [3, 4], opset=4), np.array([4, 6], dtype=np.float32))

    assert np.array_equal(import_and_compute("Add", [1, 2, 3], [4, 5, 6], opset=4), np.array([5, 7, 9], dtype=np.float32))

    assert np.array_equal(
        import_and_compute("Add", [[1, 2, 3], [4, 5, 6]], [7, 8, 9], broadcast=1, opset=4),
        np.array([[8, 10, 12], [11, 13, 15]], dtype=np.float32),
    )

    # shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
    left_operand = np.ones((2, 3, 4, 5)).astype(np.float32)
    assert np.array_equal(import_and_compute("Add", left_operand, 8, broadcast=1, opset=4), left_operand + 8)

    # shape(A) = (2, 3, 4, 5), shape(B) = (5,)
    left_operand = np.ones((2, 3, 4, 5), dtype=np.float32)
    right_operand = np.random.rand(5).astype(np.float32)
    import_and_compute("Add", left_operand, right_operand, broadcast=1, opset=4)

    # shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
    left_operand = np.ones((2, 3, 4, 5), dtype=np.float32)
    right_operand = np.random.rand(4, 5).astype(np.float32)
    assert np.array_equal(
        import_and_compute("Add", left_operand, right_operand, broadcast=1, opset=4),
        left_operand + right_operand,
    )

    # shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
    left_operand = np.ones((2, 3, 4, 5), dtype=np.float32)
    right_operand = np.random.rand(3, 4).astype(np.float32)
    assert np.array_equal(
        import_and_compute("Add", left_operand, right_operand, broadcast=1, axis=1, opset=4),
        left_operand + right_operand.reshape(1, 3, 4, 1),
    )

    # shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0
    left_operand = np.ones((2, 3, 4, 5), dtype=np.float32)
    right_operand = np.random.rand(2).astype(np.float32)
    assert np.array_equal(
        import_and_compute("Add", left_operand, right_operand, broadcast=1, axis=0, opset=4),
        left_operand + right_operand.reshape(2, 1, 1, 1),
    )


@pytest.mark.parametrize(
    ("left_shape", "right_shape"),
    [
        ((1,), (1,)),
        ((256, 256, 3), (3,)),
        ((5, 4), (1,)),
        ((5, 4), (4,)),
        ((15, 3, 5), (3, 5)),
        ((15, 3, 5), (15, 1, 5)),
        ((15, 3, 5), (3, 1)),
        ((8, 1, 6, 1), (7, 1, 5)),
    ],
)
def test_add_opset7(left_shape, right_shape):
    """Test Add-7 operator, which uses numpy-style broadcasting."""
    left_input = np.ones(left_shape)
    right_input = np.ones(right_shape)
    assert np.array_equal(import_and_compute("Add", left_input, right_input), left_input + right_input)


def test_sub():
    assert np.array_equal(import_and_compute("Sub", 20, 1), np.array(19, dtype=np.float32))

    assert np.array_equal(import_and_compute("Sub", [20], [1]), np.array([19], dtype=np.float32))

    assert np.array_equal(import_and_compute("Sub", [20, 19], [1, 2]), np.array([19, 17], dtype=np.float32))

    assert np.array_equal(
        import_and_compute("Sub", [[1, 2, 3], [4, 5, 6]], [7, 8, 9], opset=6, broadcast=1),
        np.array([[-6, -6, -6], [-3, -3, -3]], dtype=np.float32),
    )


def test_mul():
    assert np.array_equal(import_and_compute("Mul", 2, 3), np.array(6, dtype=np.float32))

    assert np.array_equal(import_and_compute("Mul", [2], [3]), np.array([6], dtype=np.float32))

    assert np.array_equal(import_and_compute("Mul", [2, 3], [4, 5]), np.array([8, 15], dtype=np.float32))

    assert np.array_equal(
        import_and_compute("Mul", [[1, 2, 3], [4, 5, 6]], [7, 8, 9], opset=6, broadcast=1),
        np.array([[7, 16, 27], [28, 40, 54]], dtype=np.float32),
    )


def test_div():
    assert np.array_equal(import_and_compute("Div", 6, 3), np.array(2, dtype=np.float32))

    assert np.array_equal(import_and_compute("Div", [6], [3]), np.array([2], dtype=np.float32))

    assert np.array_equal(import_and_compute("Div", [6, 8], [3, 2]), np.array([2, 4], dtype=np.float32))

    assert np.array_equal(
        import_and_compute("Div", [[10, 20, 30], [40, 50, 60]], [2, 5, 6], opset=6, broadcast=1),
        np.array([[5, 4, 5], [20, 10, 10]], dtype=np.float32),
    )
