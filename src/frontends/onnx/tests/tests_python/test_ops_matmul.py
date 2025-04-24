# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
import pytest

from tests.runtime import get_runtime
from tests.tests_python.utils import import_onnx_model


def make_onnx_model_for_matmul_op(input_left, input_right):
    output_shape = np.matmul(input_left, input_right).shape
    node = make_node("MatMul", ["X", "Y"], ["Z"], name="test_node")
    graph = make_graph(
        [node],
        "test_graph",
        [
            make_tensor_value_info("X", onnx.TensorProto.FLOAT, input_left.shape),
            make_tensor_value_info("Y", onnx.TensorProto.FLOAT, input_right.shape),
        ],
        [make_tensor_value_info("Z", onnx.TensorProto.FLOAT, output_shape)],
    )
    model = make_model(graph, producer_name="OpenVINO ONNX Frontend")
    return model


def import_and_compute_matmul(input_left, input_right):
    input_data_left = np.array(input_left).astype(np.float32)
    input_data_right = np.array(input_right).astype(np.float32)
    onnx_model = make_onnx_model_for_matmul_op(input_data_left, input_data_right)
    transformer = get_runtime()
    model = import_onnx_model(onnx_model)
    computation = transformer.computation(model)
    return computation(input_data_left, input_data_right)[0]


def numpy_gemm(input_a, input_b, input_c, alpha=1, beta=1, trans_a=False, trans_b=False, broadcast=False):
    input_a = np.array(input_a).astype(np.float32)
    input_b = np.array(input_b).astype(np.float32)
    input_c = np.array(input_c).astype(np.float32)

    if trans_a:
        input_a = input_a.T
    if trans_b:
        input_b = input_b.T

    return (alpha * np.dot(input_a, input_b)) + (beta * input_c)


def make_onnx_model_for_gemm_op(input_a, input_b, input_c, **kwargs):
    input_a_for_output = input_a
    input_b_for_output = input_b
    if kwargs.get("transA"):
        input_a_for_output = input_a.T
    if kwargs.get("transB"):
        input_b_for_output = input_b.T

    output_shape = np.dot(input_a_for_output, input_b_for_output).shape
    node = make_node("Gemm", ["A", "B", "C"], ["Y"], name="test_node", **kwargs)
    graph = make_graph(
        [node],
        "test_graph",
        [
            make_tensor_value_info("A", onnx.TensorProto.FLOAT, input_a.shape),
            make_tensor_value_info("B", onnx.TensorProto.FLOAT, input_b.shape),
            make_tensor_value_info("C", onnx.TensorProto.FLOAT, input_c.shape),
        ],
        [make_tensor_value_info("Y", onnx.TensorProto.FLOAT, output_shape)],
    )
    model = make_model(graph, producer_name="OpenVINO ONNX Frontend")
    return model


def import_and_compute_gemm(input_a, input_b, input_c, **kwargs):
    input_a = np.array(input_a).astype(np.float32)
    input_b = np.array(input_b).astype(np.float32)
    input_c = np.array(input_c).astype(np.float32)

    if kwargs.get("trans_a"):
        kwargs["transA"] = kwargs["trans_a"]
        del kwargs["trans_a"]

    if kwargs.get("trans_b"):
        kwargs["transB"] = kwargs["trans_b"]
        del kwargs["trans_b"]

    onnx_model = make_onnx_model_for_gemm_op(input_a, input_b, input_c, **kwargs)
    transformer = get_runtime()
    model = import_onnx_model(onnx_model)
    computation = transformer.computation(model)
    return computation(input_a, input_b, input_c)[0]


@pytest.mark.parametrize(
    ("data", "description"),
    [
        pytest.param(([1, 2], [1, 3]), "vector and vector 1"),
        (([1, 2, 3], [[4], [5], [6]]), "vector and vector 2"),
        (([[1, 2, 3]], [1, 2, 3]), "vector and vector 3"),
        (([1, 2, 3], [[4, 5], [6, 7], [8, 9]]), "vector and matrix"),
        (([[1, 2, 3], [4, 5, 6]], [[7], [8], [9]]), "matrix and vector"),
        (([[1, 2], [3, 4]], [[5, 6], [7, 8]]), "matrix and matrix 1"),
        (([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]]), "matrix and matrix 2"),
        (([[1, 2], [3, 4], [5, 6]], [[7, 8, 9], [10, 11, 12]]), "matrix and matrix 3"),
    ],
)
def test_op_matmul(data, description):
    assert np.allclose(import_and_compute_matmul(*data), np.matmul(*data))


def test_op_matmul_3d():
    # 3d tensor @ 3d tensor
    data = ([[[1, 2], [3, 4]], [[1, 2], [3, 4]]], [[[5, 6], [7, 8]], [[5, 6], [7, 8]]])
    assert np.array_equal(import_and_compute_matmul(*data), np.matmul(*data))

    data = (np.ones((5, 2, 3)), (np.ones((5, 3, 2)) + 2))
    assert np.array_equal(import_and_compute_matmul(*data), np.matmul(*data))


@pytest.mark.parametrize(
    ("data", "kwargs", "description"),
    [
        pytest.param(([1, 2], [1, 3], [1, 4]), {}, "vectors"),
        pytest.param(([1, 2], [1, 3], 1), {}, "vectors and scalar"),
        pytest.param(([1, 2], [1, 3], [1]), {}, "vectors and identity vector"),
        pytest.param(([1, 2], [1, 3], [1, 4]), {"alpha": 7.0, "beta": 9.0},
                     "vectors with alpha and beta"),
        pytest.param(([1, 2, 3, 4], [1, 3, 5, 7], [1, 4]), {"alpha": 7.0, "beta": 9.0},
                     "longer vectors with alpha and beta"),
    ],
)
def test_gemm(data, kwargs, description):
    assert np.allclose(import_and_compute_gemm(*data, **kwargs), numpy_gemm(*data, **kwargs))


@pytest.mark.parametrize(
    ("data", "kwargs", "description"),
    [
        pytest.param(([1, 2], [1, 3], [1, 4]), {"trans_a": True, "trans_b": True},
                     "vectors with trans_a/trans_b"),
        pytest.param(([[1, 2], [1, 2]], [[1, 3], [1, 3]], [4, 1]),
                     {"trans_a": True, "trans_b": True, "alpha": 7.0, "beta": 9.0},
                     "matrices and vector with trans_b and alpha/beta"),
        pytest.param(([[1, 2]], [[1, 3]], 1), {"trans_b": True, "alpha": 7.0, "beta": 9.0},
                     "matrices and scalar with trans_b and alpha/beta"),
        pytest.param(([[1], [2]], [[1], [3]], 1), {"trans_a": True, "alpha": 7.0, "beta": 9.0},
                     "matrices and scalar with trans_a and alpha/beta"),
    ],
)
def test_gemm_transpositions(data, kwargs, description):
    assert np.array_equal(import_and_compute_gemm(*data, **kwargs), numpy_gemm(*data, **kwargs))


def test_gemm_flatten():
    # input_a has a shape of (4, 1)
    data = ([[1], [2], [3], [4]], [1, 3, 5, 7], [1, 4])
    kwargs = {"alpha": 7.0, "beta": 9.0, "trans_a": True}
    assert np.array_equal(import_and_compute_gemm(*data, **kwargs), numpy_gemm(*data, **kwargs))
