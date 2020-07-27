# ******************************************************************************
# Copyright 2018-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
import numpy as np
import onnx
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

from tests.runtime import get_runtime
from tests.test_onnx.utils import import_onnx_model


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
    model = make_model(graph, producer_name="ngraph ONNXImporter")
    return model


def import_and_compute_matmul(input_left, input_right):
    input_data_left = np.array(input_left)
    input_data_right = np.array(input_right)
    onnx_model = make_onnx_model_for_matmul_op(input_data_left, input_data_right)
    transformer = get_runtime()
    ng_model_function = import_onnx_model(onnx_model)
    computation = transformer.computation(ng_model_function)
    return computation(input_data_left, input_data_right)[0]


def numpy_gemm(input_a, input_b, input_c, alpha=1, beta=1, trans_a=False, trans_b=False, broadcast=False):
    input_a, input_b, input_c = np.array(input_a), np.array(input_b), np.array(input_c)
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
    model = make_model(graph, producer_name="ngraph ONNXImporter")
    return model


def import_and_compute_gemm(input_a, input_b, input_c, **kwargs):
    input_a, input_b, input_c = np.array(input_a), np.array(input_b), np.array(input_c)

    if kwargs.get("trans_a"):
        kwargs["transA"] = kwargs["trans_a"]
        del kwargs["trans_a"]

    if kwargs.get("trans_b"):
        kwargs["transB"] = kwargs["trans_b"]
        del kwargs["trans_b"]

    onnx_model = make_onnx_model_for_gemm_op(input_a, input_b, input_c, **kwargs)
    transformer = get_runtime()
    ng_model_function = import_onnx_model(onnx_model)
    computation = transformer.computation(ng_model_function)
    return computation(input_a, input_b, input_c)[0]


def test_op_matmul():
    # vector @ vector
    data = ([1, 2], [1, 3])
    assert np.array_equal(import_and_compute_matmul(*data), np.matmul(*data))

    data = ([1, 2, 3], [[4], [5], [6]])
    assert np.array_equal(import_and_compute_matmul(*data), np.matmul(*data))

    data = ([[1, 2, 3]], [1, 2, 3])
    assert np.array_equal(import_and_compute_matmul(*data), np.matmul(*data))

    # vector @ matrix
    data = ([1, 2, 3], [[4, 5], [6, 7], [8, 9]])
    assert np.array_equal(import_and_compute_matmul(*data), np.matmul(*data))

    # matrix @ vector
    data = ([[1, 2, 3], [4, 5, 6]], [[7], [8], [9]])
    assert np.array_equal(import_and_compute_matmul(*data), np.matmul(*data))

    # matrix @ matrix
    data = ([[1, 2], [3, 4]], [[5, 6], [7, 8]])
    assert np.array_equal(import_and_compute_matmul(*data), np.matmul(*data))

    data = ([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]])
    assert np.array_equal(import_and_compute_matmul(*data), np.matmul(*data))

    data = ([[1, 2], [3, 4], [5, 6]], [[7, 8, 9], [10, 11, 12]])
    assert np.array_equal(import_and_compute_matmul(*data), np.matmul(*data))


def test_op_matmul_3d():
    # 3d tensor @ 3d tensor
    data = ([[[1, 2], [3, 4]], [[1, 2], [3, 4]]], [[[5, 6], [7, 8]], [[5, 6], [7, 8]]])
    assert np.array_equal(import_and_compute_matmul(*data), np.matmul(*data))

    data = (np.ones((5, 2, 3)), (np.ones((5, 3, 2)) + 2))
    assert np.array_equal(import_and_compute_matmul(*data), np.matmul(*data))


def test_gemm():
    data = ([1, 2], [1, 3], [1, 4])
    assert np.array_equal(import_and_compute_gemm(*data), numpy_gemm(*data))

    data = ([1, 2], [1, 3], 1)
    assert np.array_equal(import_and_compute_gemm(*data), numpy_gemm(*data))

    data = ([1, 2], [1, 3], [1])
    assert np.array_equal(import_and_compute_gemm(*data), numpy_gemm(*data))

    data = ([1, 2], [1, 3], [1, 4])
    kwargs = {"alpha": 7, "beta": 9}
    assert np.array_equal(import_and_compute_gemm(*data, **kwargs), numpy_gemm(*data, **kwargs))

    data = ([1, 2, 3, 4], [1, 3, 5, 7], [1, 4])
    kwargs = {"alpha": 7, "beta": 9}
    assert np.array_equal(import_and_compute_gemm(*data, **kwargs), numpy_gemm(*data, **kwargs))


def test_gemm_transpositions():
    data = ([1, 2], [1, 3], [1, 4])
    kwargs = {"trans_a": True, "trans_b": True}
    assert np.array_equal(import_and_compute_gemm(*data, **kwargs), numpy_gemm(*data, **kwargs))

    data = ([[1, 2], [1, 2]], [[1, 3], [1, 3]], [4, 1])
    kwargs = {"trans_a": True, "trans_b": True, "alpha": 7, "beta": 9}
    assert np.array_equal(import_and_compute_gemm(*data, **kwargs), numpy_gemm(*data, **kwargs))

    data = ([[1, 2]], [[1, 3]], 1)
    kwargs = {"trans_b": True, "alpha": 7, "beta": 9}
    assert np.array_equal(import_and_compute_gemm(*data, **kwargs), numpy_gemm(*data, **kwargs))

    data = ([[1], [2]], [[1], [3]], 1)
    kwargs = {"trans_a": True, "alpha": 7, "beta": 9}
    assert np.array_equal(import_and_compute_gemm(*data, **kwargs), numpy_gemm(*data, **kwargs))


def test_gemm_flatten():
    # input_a.shape is (4,1,1)
    data = ([[[1]], [[2]], [[3]], [[4]]], [1, 3, 5, 7], [1, 4])
    kwargs = {"alpha": 7, "beta": 9}
    assert np.array_equal(import_and_compute_gemm(*data, **kwargs), numpy_gemm(*data, **kwargs))
