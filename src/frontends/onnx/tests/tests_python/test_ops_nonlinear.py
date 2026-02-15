# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx
import pytest

from tests.tests_python.utils import run_node


def import_and_compute(op_type, input_data, **node_attrs):
    data_inputs = [np.array(input_data)]
    node = onnx.helper.make_node(op_type, inputs=["x"], outputs=["y"], **node_attrs)
    return run_node(node, data_inputs).pop()


def assert_onnx_import_equals_callable(onnx_op_type, python_function, data, **kwargs):
    data = np.array(data, dtype=np.float32)
    assert np.allclose(import_and_compute(onnx_op_type, data, **kwargs), python_function(data, **kwargs))


def test_sigmoid():
    def sigmoid(value):
        return 1 / (1 + np.exp(-value))

    assert_onnx_import_equals_callable("Sigmoid", sigmoid, [-2, -1.0, 0.0, 1.0, 2.0])
    assert_onnx_import_equals_callable("Sigmoid", sigmoid, [0.0])
    assert_onnx_import_equals_callable("Sigmoid", sigmoid, [-2, -1.0, 0.0, 1.0, 2.0])


def test_tanh():
    assert_onnx_import_equals_callable("Tanh", np.tanh, [-2, -1.0, 0.0, 1.0, 2.0])
    assert_onnx_import_equals_callable("Tanh", np.tanh, [0.0])
    assert_onnx_import_equals_callable("Tanh", np.tanh, [-2, -1.0, 0.0, 1.0, 2.0])


def test_relu():
    def relu(value):
        return np.maximum(value, 0)

    assert_onnx_import_equals_callable("Relu", relu, [-2, -1.0, 0.0, 1.0, 2.0])
    assert_onnx_import_equals_callable("Relu", relu, [0.0])
    assert_onnx_import_equals_callable("Relu", relu, [-0.9, -0.8, -0.7, -0.4, -0.3, -0.2, -0.1])
    assert_onnx_import_equals_callable("Relu", relu, [[1, 2, 3], [4, 5, 6]])
    assert_onnx_import_equals_callable("Relu", relu, [[-3, -2, -1], [1, 2, 3]])


def test_leaky_relu():
    def leaky_relu(value, alpha=0.01):
        return np.maximum(alpha * value, value)

    assert_onnx_import_equals_callable("LeakyRelu", leaky_relu, [-2, -1.0, 0.0, 1.0, 2.0], alpha=0.5)
    assert_onnx_import_equals_callable("LeakyRelu", leaky_relu, [0.0])
    assert_onnx_import_equals_callable("LeakyRelu", leaky_relu, [-0.9, -0.8, -0.7, -0.4, -0.3, -0.2, -0.1], alpha=1.0)
    assert_onnx_import_equals_callable("LeakyRelu", leaky_relu, [[1, 2, 3], [4, 5, 6]], alpha=0.2)
    assert_onnx_import_equals_callable("LeakyRelu", leaky_relu, [[-3, -2, -1], [1, 2, 3]])


@pytest.mark.parametrize(
    ("value", "slope"),
    [
        ([-2, -1.0, 0.0, 1.0, 2.0], 0.5),
        ([0.0], 1),
        ([-0.9, -0.8, -0.7, -0.4, -0.3, -0.2, -0.1], 1),
        ([[1, 2, 3], [4, 5, 6]], 0.5),
        ([[-3, -2, -1], [1, 2, 3]], 1),
    ],
)
def test_parametric_relu(value, slope):
    def parametic_relu(value, slope):
        return np.where(value < 0, slope * value, value)

    value, slope = np.array(value).astype(np.float32), np.array(slope).astype(np.float32)
    expected_output = parametic_relu(value, slope)
    node = onnx.helper.make_node("PRelu", inputs=["x", "slope"], outputs=["y"])
    output = run_node(node, [value, slope]).pop()
    assert np.allclose(output, expected_output)


def test_selu():
    # f(x) = gamma * (alpha * exp(x) - alpha) for x <= 0, y = gamma * x for x > 0
    def selu(value, alpha=1.67326319217681884765625, gamma=1.05070102214813232421875):
        return np.where(value <= 0, gamma * (alpha * np.exp(value) - alpha), gamma * value)

    assert_onnx_import_equals_callable("Selu", selu, [-2, -1.0, 0.0, 1.0, 2.0])
    assert_onnx_import_equals_callable("Selu", selu, [0.0])
    assert_onnx_import_equals_callable("Selu", selu, [-0.9, -0.8, -0.7, -0.4, -0.3, -0.2, -0.1])
    assert_onnx_import_equals_callable("Selu", selu, [[1, 2, 3], [4, 5, 6]])
    assert_onnx_import_equals_callable("Selu", selu, [-2, -1.0, 0.0, 1.0, 2.0], gamma=0.5, alpha=0.5)


@pytest.mark.parametrize(
    ("data", "alpha_value"),
    [
        pytest.param([-2, -1.0, 0.0, 1.0, 2.0], 1.0),
        pytest.param([0.0], 1.0),
        pytest.param([-0.9, -0.8, -0.7, -0.4, -0.3, -0.2, -0.1], 1.0),
        pytest.param([[1, 2, 3], [4, 5, 6]], 1.0),
        pytest.param([-2, -1.0, 0.0, 1.0, 2.0], 0.5),
    ],
)
def test_elu(data, alpha_value):
    # f(x) = alpha * (exp(x) - 1) for x < 0, f(x) = x for x >= 0
    def elu(value, alpha):
        return np.where(value < 0, alpha * (np.exp(value) - 1), value)

    assert_onnx_import_equals_callable("Elu", elu, data, alpha=alpha_value)
