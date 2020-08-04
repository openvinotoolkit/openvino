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
import pytest
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

from tests.runtime import get_runtime
from tests.test_onnx.utils import (all_arrays_equal,
                                   get_node_model,
                                   import_onnx_model,
                                   run_model,
                                   run_node,
                                   xfail_issue_35926,
                                   xfail_issue_35927
                                   )


@xfail_issue_35926
def test_reshape():
    input_data = np.arange(2560).reshape([16, 4, 4, 10])
    reshape_node = onnx.helper.make_node("Reshape", inputs=["x"], outputs=["y"], shape=(256, 10))
    expected_output = input_data.reshape([256, 10])

    ng_results = run_node(reshape_node, [input_data], opset_version=4)
    assert np.array_equal(ng_results, [expected_output])


def test_reshape_opset5():
    original_shape = [2, 3, 4]
    test_cases = {
        "reordered_dims": np.array([4, 2, 3], dtype=np.int64),
        "reduced_dims": np.array([3, 8], dtype=np.int64),
        "extended_dims": np.array([3, 2, 2, 2], dtype=np.int64),
        "one_dim": np.array([24], dtype=np.int64),
        "negative_dim": np.array([6, -1, 2], dtype=np.int64),
    }
    input_data = np.random.random_sample(original_shape).astype(np.float32)

    for _, shape in test_cases.items():
        const_node = make_node(
            "Constant",
            inputs=[],
            outputs=["const_shape"],
            value=onnx.helper.make_tensor(
                name="const_tensor", data_type=onnx.TensorProto.INT64, dims=shape.shape, vals=shape.flatten()
            ),
        )
        reshape_node = onnx.helper.make_node("Reshape", inputs=["data", "const_shape"], outputs=["reshaped"])

        graph = make_graph(
            [const_node, reshape_node],
            "test_graph",
            [make_tensor_value_info("data", onnx.TensorProto.FLOAT, input_data.shape)],
            [make_tensor_value_info("reshaped", onnx.TensorProto.FLOAT, ())],
        )

        model = make_model(graph, producer_name="ngraph ONNX Importer")
        model.opset_import[0].version = 5
        ng_model_function = import_onnx_model(model)
        runtime = get_runtime()
        computation = runtime.computation(ng_model_function)
        ng_results = computation(input_data)
        expected_output = np.reshape(input_data, shape)
        assert np.array_equal(ng_results[0], expected_output)


@xfail_issue_35926
def test_reshape_opset5_param_err():
    original_shape = [2, 3, 4]
    output_shape = np.array([4, 2, 3], dtype=np.int64)
    input_data = np.random.random_sample(original_shape).astype(np.float32)
    reshape_node = onnx.helper.make_node("Reshape", inputs=["x", "y"], outputs=["z"])
    ng_result = run_node(reshape_node, [input_data, output_shape], opset_version=5)
    assert ng_result[0].shape == output_shape


@xfail_issue_35926
@pytest.mark.parametrize(
    "axis,expected_output",
    [
        (0, np.arange(120).reshape(1, 120)),
        (1, np.arange(120).reshape(2, 60)),
        (2, np.arange(120).reshape(6, 20)),
        (3, np.arange(120).reshape(24, 5)),
        (4, np.arange(120).reshape(120, 1)),
    ],
)
def test_flatten(axis, expected_output):
    data = np.arange(120).reshape([2, 3, 4, 5])
    node = onnx.helper.make_node("Flatten", inputs=["x"], outputs=["y"], axis=axis)
    ng_results = run_node(node, [data])
    assert np.array_equal(ng_results, [expected_output])


def test_flatten_exception():
    data = np.arange(120).reshape([2, 3, 4, 5])
    node = onnx.helper.make_node("Flatten", inputs=["x"], outputs=["y"], axis=5)

    with pytest.raises(RuntimeError):
        run_node(node, [data])


@xfail_issue_35926
def test_transpose():
    data = np.arange(120).reshape([2, 3, 4, 5])

    node = onnx.helper.make_node("Transpose", inputs=["x"], outputs=["y"])
    expected_output = data.T
    ng_results = run_node(node, [data])
    assert np.array_equal(ng_results, [expected_output])

    node = onnx.helper.make_node("Transpose", inputs=["x"], outputs=["y"], perm=(3, 1, 0, 2))
    expected_output = np.transpose(data, axes=(3, 1, 0, 2))
    ng_results = run_node(node, [data])
    assert np.array_equal(ng_results, [expected_output])


@xfail_issue_35927
def test_slice_opset1():
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

    expected_output = np.array([[5, 6, 7]])
    model = get_node_model("Slice", data, axes=[0, 1], starts=[1, 0], ends=[2, 3])
    ng_results = run_model(model, [data])
    assert np.array_equal(ng_results, [expected_output])

    expected_output = np.array([[2, 3, 4]])
    model = get_node_model("Slice", data, starts=[0, 1], ends=[-1, 1000])
    ng_results = run_model(model, [data])
    assert np.array_equal(ng_results, [expected_output])

    data = np.random.randn(20, 10, 5).astype(np.float32)
    expected_output = data[0:3, 0:10]
    model = get_node_model("Slice", data, axes=[0, 1], starts=[0, 0], ends=[3, 10])
    ng_results = run_model(model, [data])
    assert np.array_equal(ng_results, [expected_output])

    # default axes
    data = np.random.randn(20, 10, 5).astype(np.float32)
    expected_output = data[:, :, 3:4]
    model = get_node_model("Slice", data, starts=[0, 0, 3], ends=[20, 10, 4])
    ng_results = run_model(model, [data])
    assert np.array_equal(ng_results, [expected_output])

    # end out of bounds
    data = np.random.randn(20, 10, 5).astype(np.float32)
    expected_output = data[:, 1:1000]
    model = get_node_model("Slice", data, axes=[1], starts=[1], ends=[1000])
    ng_results = run_model(model, [data])
    assert np.array_equal(ng_results, [expected_output])

    # negative value
    data = np.random.randn(20, 10, 5).astype(np.float32)
    expected_output = data[:, 0:-1]
    model = get_node_model("Slice", data, axes=[1], starts=[0], ends=[-1])
    ng_results = run_model(model, [data])
    assert np.array_equal(ng_results, [expected_output])

    # start ouf of bounds
    data = np.random.randn(20, 10, 5).astype(np.float32)
    expected_output = data[:, 1000:1000]
    model = get_node_model("Slice", data, axes=[1], starts=[1000], ends=[1000])
    ng_results = run_model(model, [data])
    assert np.array_equal(ng_results, [expected_output])


@xfail_issue_35926
def test_concat():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6]])

    node = onnx.helper.make_node("Concat", inputs=["x"], outputs=["z"], axis=0)
    ng_results = run_node(node, [a])
    assert np.array_equal(ng_results, [a])

    expected_output = np.concatenate((a, b), axis=0)
    node = onnx.helper.make_node("Concat", inputs=["x", "y"], outputs=["z"], axis=0)
    ng_results = run_node(node, [a, b])
    assert np.array_equal(ng_results, [expected_output])

    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6]]).T
    expected_output = np.concatenate((a, b), axis=1)
    node = onnx.helper.make_node("Concat", inputs=["x", "y"], outputs=["z"], axis=1)
    ng_results = run_node(node, [a, b])
    assert np.array_equal(ng_results, [expected_output])

    test_cases = {
        "1d": ([1, 2], [3, 4]),
        "2d": ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
        "3d": ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]),
    }

    for _, values in test_cases.items():
        values = [np.asarray(v) for v in values]
        for i in range(len(values[0].shape)):
            in_args = ["value" + str(k) for k in range(len(values))]
            node = onnx.helper.make_node("Concat", inputs=list(in_args), outputs=["output"], axis=i,)
            expected_output = np.concatenate(values, i)
            ng_results = run_node(node, list(values))
            assert np.array_equal(ng_results, [expected_output])


@xfail_issue_35926
def test_squeeze():
    data = np.arange(6).reshape([1, 2, 3, 1])
    expected_output = data.reshape([2, 3])

    node = onnx.helper.make_node("Squeeze", inputs=["x"], outputs=["y"], axes=[0, 3])
    ng_results = run_node(node, [data])
    assert np.array_equal(ng_results, [expected_output])

    data = np.random.randn(1, 3, 4, 5).astype(np.float32)
    expected_output = np.squeeze(data, axis=0)
    node = onnx.helper.make_node("Squeeze", inputs=["x"], outputs=["y"], axes=[0])
    ng_results = run_node(node, [data])
    assert np.array_equal(ng_results, [expected_output])


def test_unsqueeze():
    data = np.random.randn(3, 4, 5).astype(np.float32)
    expected_output = np.expand_dims(data, axis=0)
    node = onnx.helper.make_node("Unsqueeze", inputs=["x"], outputs=["y"], axes=[0])
    ng_results = run_node(node, [data])
    assert np.array_equal(ng_results, [expected_output])

    expected_output = np.reshape(data, [1, 3, 4, 5, 1])
    node = onnx.helper.make_node("Unsqueeze", inputs=["x"], outputs=["y"], axes=[0, 4])
    ng_results = run_node(node, [data])
    assert np.array_equal(ng_results, [expected_output])

    expected_output = np.reshape(data, [1, 3, 1, 4, 5])
    node = onnx.helper.make_node("Unsqueeze", inputs=["x"], outputs=["y"], axes=[0, 2])
    ng_results = run_node(node, [data])
    assert np.array_equal(ng_results, [expected_output])


@xfail_issue_35926
@pytest.mark.parametrize(
    "node, expected_output",
    [
        # Split into 2 equal parts along axis=0
        (
            onnx.helper.make_node("Split", inputs=["x"], outputs=["y", "z"], axis=0),
            [np.array([[0, 1, 2, 3]]), np.array([[4, 5, 6, 7]])],
        ),
        # Default, split along axis=0 into 2 equal parts
        (
            onnx.helper.make_node("Split", inputs=["x"], outputs=["y", "z"]),
            [np.array([[0, 1, 2, 3]]), np.array([[4, 5, 6, 7]])],
        ),
        # Split into 2 equal parts along axis=1
        (
            onnx.helper.make_node("Split", inputs=["x"], outputs=["a", "b"], axis=1),
            [np.array([[0, 1], [4, 5]]), np.array([[2, 3], [6, 7]])],
        ),
        # Split into 4 equal parts along axis=1
        (
            onnx.helper.make_node("Split", inputs=["x"], outputs=["a", "b", "c", "d"], axis=1),
            [np.array([[0], [4]]), np.array([[1], [5]]), np.array([[2], [6]]), np.array([[3], [7]])],
        ),
        # Split into 2 unequal parts along axis=1
        (
            onnx.helper.make_node("Split", inputs=["x"], outputs=["a", "b"], axis=1, split=(3, 1)),
            [np.array([[0, 1, 2], [4, 5, 6]]), np.array([[3], [7]])],
        ),
    ],
)
def test_split_2d(node, expected_output):
    data = np.arange(8).reshape(2, 4)
    ng_results = run_node(node, [data])
    assert all_arrays_equal(ng_results, expected_output)


def test_split_1d():
    # 1D
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32)

    node = onnx.helper.make_node("Split", inputs=["input"], outputs=["z", "w"], axis=0)
    expected_outputs = [
        np.array([1.0, 2.0, 3.0]).astype(np.float32),
        np.array([4.0, 5.0, 6.0]).astype(np.float32),
    ]
    ng_results = run_node(node, [data])
    assert all_arrays_equal(ng_results, expected_outputs)

    node = onnx.helper.make_node("Split", inputs=["input"], outputs=["y", "z", "w"], axis=0, split=[2, 3, 1])
    expected_outputs = [
        np.array([1.0, 2.0]).astype(np.float32),
        np.array([3.0, 4.0, 5.0]).astype(np.float32),
        np.array([6.0]).astype(np.float32),
    ]
    ng_results = run_node(node, [data])
    assert all_arrays_equal(ng_results, expected_outputs)

    # Default values
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32)

    node = onnx.helper.make_node("Split", inputs=["input"], outputs=["y", "z", "w"])
    expected_outputs = [
        np.array([1.0, 2.0]).astype(np.float32),
        np.array([3.0, 4.0]).astype(np.float32),
        np.array([5.0, 6.0]).astype(np.float32),
    ]
    ng_results = run_node(node, [data])
    assert all_arrays_equal(ng_results, expected_outputs)

    node = onnx.helper.make_node("Split", inputs=["input"], outputs=["y", "z"], split=[2, 4])
    expected_outputs = [
        np.array([1.0, 2.0]).astype(np.float32),
        np.array([3.0, 4.0, 5.0, 6.0]).astype(np.float32),
    ]
    ng_results = run_node(node, [data])
    assert all_arrays_equal(ng_results, expected_outputs)


def test_depth_to_space():
    b, c, h, w = shape = (2, 8, 3, 3)
    blocksize = 2
    data = np.random.random_sample(shape).astype(np.float32)
    tmp = np.reshape(data, [b, blocksize, blocksize, c // (blocksize ** 2), h, w])
    tmp = np.transpose(tmp, [0, 3, 4, 1, 5, 2])
    expected_output = np.reshape(tmp, [b, c // (blocksize ** 2), h * blocksize, w * blocksize])

    node = onnx.helper.make_node("DepthToSpace", inputs=["x"], outputs=["y"], blocksize=blocksize)
    ng_results = run_node(node, [data])
    assert np.array_equal(ng_results, [expected_output])

    # (1, 4, 2, 3) input tensor
    data = np.array(
        [
            [
                [[0, 1, 2], [3, 4, 5]],
                [[6, 7, 8], [9, 10, 11]],
                [[12, 13, 14], [15, 16, 17]],
                [[18, 19, 20], [21, 22, 23]],
            ]
        ]
    ).astype(np.float32)
    # (1, 1, 4, 6) output tensor
    expected_output = np.array(
        [[[[0, 6, 1, 7, 2, 8], [12, 18, 13, 19, 14, 20], [3, 9, 4, 10, 5, 11], [15, 21, 16, 22, 17, 23]]]]
    ).astype(np.float32)

    ng_results = run_node(node, [data])
    assert np.array_equal(ng_results, [expected_output])
