# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx
import pytest
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

from tests.runtime import get_runtime
from tests.tests_python.utils import (
    all_arrays_equal,
    get_node_model,
    import_onnx_model,
    run_model,
    run_node,
)
from tests import (xfail_issue_44858,
                   xfail_dynamic_rank)


def test_reshape():
    input_data = np.arange(2560, dtype=np.int32).reshape([16, 4, 4, 10])
    reshape_node = onnx.helper.make_node(
        "Reshape", inputs=["x"], outputs=["y"], shape=(256, 10),
    )
    expected_output = input_data.reshape([256, 10])

    graph_results = run_node(reshape_node, [input_data], opset_version=4)
    assert np.array_equal(graph_results, [expected_output])


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
                name="const_tensor",
                data_type=onnx.TensorProto.INT64,
                dims=shape.shape,
                vals=shape.flatten(),
            ),
        )
        reshape_node = onnx.helper.make_node(
            "Reshape", inputs=["data", "const_shape"], outputs=["reshaped"],
        )

        graph = make_graph(
            [const_node, reshape_node],
            "test_graph",
            [make_tensor_value_info("data", onnx.TensorProto.FLOAT, input_data.shape)],
            [make_tensor_value_info("reshaped", onnx.TensorProto.FLOAT, ())],
        )

        model = make_model(graph, producer_name="OpenVINO ONNX Frontend")
        model.opset_import[0].version = 5
        graph_model_function = import_onnx_model(model)
        runtime = get_runtime()
        computation = runtime.computation(graph_model_function)
        graph_results = computation(input_data)
        expected_output = np.reshape(input_data, shape)
        assert np.array_equal(graph_results[0], expected_output)


@pytest.mark.xfail(reason="RuntimeError: Reshape z has dynamic second input!")
def test_reshape_opset5_param_err():
    original_shape = [2, 3, 4]
    output_shape = np.array([4, 2, 3], dtype=np.int32)
    input_data = np.random.random_sample(original_shape).astype(np.float32)
    reshape_node = onnx.helper.make_node("Reshape", inputs=["x", "y"], outputs=["z"])
    graph_result = run_node(reshape_node, [input_data, output_shape], opset_version=5)
    assert graph_result[0].shape == output_shape


@pytest.mark.parametrize(
    ("axis", "expected_output"),
    [
        (0, np.arange(120).reshape(1, 120)),
        (1, np.arange(120).reshape(2, 60)),
        (2, np.arange(120).reshape(6, 20)),
        (3, np.arange(120).reshape(24, 5)),
        (4, np.arange(120).reshape(120, 1)),
    ],
)
def test_flatten(axis, expected_output):
    data = np.arange(120, dtype=np.int32).reshape([2, 3, 4, 5])
    node = onnx.helper.make_node("Flatten", inputs=["x"], outputs=["y"], axis=axis)
    graph_results = run_node(node, [data])
    assert np.array_equal(graph_results, [expected_output])


def test_flatten_exception():
    data = np.arange(120).reshape([2, 3, 4, 5])
    node = onnx.helper.make_node("Flatten", inputs=["x"], outputs=["y"], axis=5)

    with pytest.raises(RuntimeError):
        run_node(node, [data])


def test_transpose():
    data = np.arange(120, dtype=np.int32).reshape([2, 3, 4, 5])

    node = onnx.helper.make_node("Transpose", inputs=["x"], outputs=["y"])
    expected_output = data.T
    graph_results = run_node(node, [data])
    assert np.array_equal(graph_results, [expected_output])

    node = onnx.helper.make_node(
        "Transpose", inputs=["x"], outputs=["y"], perm=(3, 1, 0, 2),
    )
    expected_output = np.transpose(data, axes=(3, 1, 0, 2))
    graph_results = run_node(node, [data])
    assert np.array_equal(graph_results, [expected_output])


def test_slice_opset1():
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

    expected_output = np.array([[5, 6, 7]])
    model = get_node_model("Slice", data, axes=[0, 1], starts=[1, 0], ends=[2, 3])
    graph_results = run_model(model, [data])
    assert np.array_equal(graph_results, [expected_output])

    expected_output = np.array([[2, 3, 4]])
    model = get_node_model("Slice", data, starts=[0, 1], ends=[-1, 1000])
    graph_results = run_model(model, [data])
    assert np.array_equal(graph_results, [expected_output])

    data = np.random.randn(20, 10, 5).astype(np.float32)
    expected_output = data[0:3, 0:10]
    model = get_node_model("Slice", data, axes=[0, 1], starts=[0, 0], ends=[3, 10])
    graph_results = run_model(model, [data])
    assert np.array_equal(graph_results, [expected_output])

    # default axes
    data = np.random.randn(20, 10, 5).astype(np.float32)
    expected_output = data[:, :, 3:4]
    model = get_node_model("Slice", data, starts=[0, 0, 3], ends=[20, 10, 4])
    graph_results = run_model(model, [data])
    assert np.array_equal(graph_results, [expected_output])

    # end out of bounds
    data = np.random.randn(20, 10, 5).astype(np.float32)
    expected_output = data[:, 1:1000]
    model = get_node_model("Slice", data, axes=[1], starts=[1], ends=[1000])
    graph_results = run_model(model, [data])
    assert np.array_equal(graph_results, [expected_output])

    # negative value
    data = np.random.randn(20, 10, 5).astype(np.float32)
    expected_output = data[:, 0:-1]
    model = get_node_model("Slice", data, axes=[1], starts=[0], ends=[-1])
    graph_results = run_model(model, [data])
    assert np.array_equal(graph_results, [expected_output])

    # start ouf of bounds
    data = np.random.randn(20, 10, 5).astype(np.float32)
    expected_output = data[:, 1000:1000]
    model = get_node_model("Slice", data, axes=[1], starts=[1000], ends=[1000])
    graph_results = run_model(model, [data])
    assert np.array_equal(graph_results, [expected_output])


def test_concat():
    input_1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_2 = np.array([[5, 6]], dtype=np.int32)

    node = onnx.helper.make_node("Concat", inputs=["x"], outputs=["z"], axis=0)
    graph_results = run_node(node, [input_1])
    assert np.array_equal(graph_results, [input_1])

    expected_output = np.concatenate((input_1, input_2), axis=0)
    node = onnx.helper.make_node("Concat", inputs=["x", "y"], outputs=["z"], axis=0)
    graph_results = run_node(node, [input_1, input_2])
    assert np.array_equal(graph_results, [expected_output])

    input_1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_2 = np.array([[5, 6]], dtype=np.int32).T
    expected_output = np.concatenate((input_1, input_2), axis=1)
    node = onnx.helper.make_node("Concat", inputs=["x", "y"], outputs=["z"], axis=1)
    graph_results = run_node(node, [input_1, input_2])
    assert np.array_equal(graph_results, [expected_output])

    test_cases = {
        "1d": ([1, 2], [3, 4]),
        "2d": ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
        "3d": (
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            [[[9, 10], [11, 12]], [[13, 14], [15, 16]]],
        ),
    }

    for _, values in test_cases.items():
        values = [np.asarray(v) for v in values]
        for i in range(len(values[0].shape)):
            in_args = ["value" + str(k) for k in range(len(values))]
            node = onnx.helper.make_node(
                "Concat",
                inputs=list(in_args),
                outputs=["output"],
                axis=i,
            )
            expected_output = np.concatenate(values, i)
            graph_results = run_node(node, np.array(values, dtype=np.int32))
            assert np.array_equal(graph_results, [expected_output])


def test_squeeze():
    data = np.arange(6, dtype=np.int32).reshape([1, 2, 1, 3, 1])
    expected_output = data.reshape([1, 2, 3, 1])
    axes = np.array([2]).astype(np.int64)
    node = onnx.helper.make_node("Squeeze", inputs=["x", "axes"], outputs=["y"])
    graph_results = run_node(node, [data, axes])
    assert np.array_equal(graph_results, [expected_output])

    data = np.arange(6, dtype=np.int32).reshape([2, 3, 1])
    expected_output = data.reshape([2, 3])
    axes = np.array([-1]).astype(np.int64)
    node = onnx.helper.make_node("Squeeze", inputs=["x", "axes"], outputs=["y"])
    graph_results = run_node(node, [data, axes])
    assert np.array_equal(graph_results, [expected_output])

    data = np.random.randn(1, 3, 4, 5).astype(np.float32)
    expected_output = np.squeeze(data, axis=0)
    axes = np.array([0]).astype(np.int64)
    node = onnx.helper.make_node("Squeeze", inputs=["x", "axes"], outputs=["y"])
    graph_results = run_node(node, [data, axes])
    assert np.array_equal(graph_results, [expected_output])

    data = np.arange(6, dtype=np.int32).reshape([1, 2, 1, 3, 1])
    expected_output = data.reshape([1, 2, 3, 1])
    axes = np.array([2]).astype(np.int64)
    node = onnx.helper.make_node("Squeeze", inputs=["x", "axes"], outputs=["y"])
    graph_results = run_node(node, [data, axes])
    assert np.array_equal(graph_results, [expected_output])


@xfail_dynamic_rank
def test_squeeze_dyn_rank():
    data = np.arange(6, dtype=np.int32).reshape([1, 2, 3, 1])
    expected_output = data.reshape([2, 3])
    axes = np.array([0, 3]).astype(np.int64)
    node = onnx.helper.make_node("Squeeze", inputs=["x", "axes"], outputs=["y"])
    graph_results = run_node(node, [data, axes])
    assert np.array_equal(graph_results, [expected_output])

    data = np.arange(6, dtype=np.int32).reshape([1, 2, 3, 1])
    expected_output = data.reshape([2, 3])
    axes = np.array([0, 0]).astype(np.int64)
    node = onnx.helper.make_node("Squeeze", inputs=["x", "axes"], outputs=["y"])
    graph_results = run_node(node, [data, axes])
    assert np.array_equal(graph_results, [expected_output])

    data = np.arange(6, dtype=np.int32).reshape([1, 2, 1, 3, 1])
    expected_output = data.reshape([1, 2, 3])
    axes = np.array([-3, -1]).astype(np.int64)
    node = onnx.helper.make_node("Squeeze", inputs=["x", "axes"], outputs=["y"])
    graph_results = run_node(node, [data, axes])
    assert np.array_equal(graph_results, [expected_output])

    data = np.arange(6, dtype=np.int32).reshape([2, 1, 3])
    expected_output = data.reshape([2, 3])
    axes = np.array([1, 1]).astype(np.int64)
    node = onnx.helper.make_node("Squeeze", inputs=["x", "axes"], outputs=["y"])
    graph_results = run_node(node, [data, axes])
    assert np.array_equal(graph_results, [expected_output])


@xfail_issue_44858
def test_unsqueeze():
    data = np.random.randn(3, 4, 5).astype(np.float32)
    expected_output = np.expand_dims(data, axis=0)
    axes = np.array([0]).astype(np.int64)
    node = onnx.helper.make_node("Unsqueeze", inputs=["x", "axes"], outputs=["y"])
    graph_results = run_node(node, [data, axes])
    assert np.array_equal(graph_results, [expected_output])

    expected_output = np.reshape(data, [1, 3, 4, 5, 1])
    axes = np.array([0, 4]).astype(np.int64)
    node = onnx.helper.make_node("Unsqueeze", inputs=["x", "axes"], outputs=["y"])
    graph_results = run_node(node, [data, axes])
    assert np.array_equal(graph_results, [expected_output])

    expected_output = np.reshape(data, [1, 3, 1, 4, 5])
    axes = np.array([0, 2]).astype(np.int64)
    node = onnx.helper.make_node("Unsqueeze", inputs=["x", "axes"], outputs=["y"])
    graph_results = run_node(node, [data, axes])
    assert np.array_equal(graph_results, [expected_output])


@pytest.mark.parametrize(
    ("node", "expected_output"),
    [
        # Split into 2 equal parts along axis=0
        (
            onnx.helper.make_node("Split", inputs=["x"], outputs=["y", "z"], axis=0),
            [
                np.array([[0, 1, 2, 3]], dtype=np.int32),
                np.array([[4, 5, 6, 7]], dtype=np.int32),
            ],
        ),
        # Default, split along axis=0 into 2 equal parts
        (
            onnx.helper.make_node("Split", inputs=["x"], outputs=["y", "z"]),
            [
                np.array([[0, 1, 2, 3]], dtype=np.int32),
                np.array([[4, 5, 6, 7]], dtype=np.int32),
            ],
        ),
        # Split into 2 equal parts along axis=1
        (
            onnx.helper.make_node("Split", inputs=["x"], outputs=["a", "b"], axis=1),
            [
                np.array([[0, 1], [4, 5]], dtype=np.int32),
                np.array([[2, 3], [6, 7]], dtype=np.int32),
            ],
        ),
        # Split into 4 equal parts along axis=1
        (
            onnx.helper.make_node(
                "Split", inputs=["x"], outputs=["a", "b", "c", "d"], axis=1,
            ),
            [
                np.array([[0], [4]], dtype=np.int32),
                np.array([[1], [5]], dtype=np.int32),
                np.array([[2], [6]], dtype=np.int32),
                np.array([[3], [7]], dtype=np.int32),
            ],
        ),
    ],
)
def test_split_2d(node, expected_output):
    data = np.arange(8, dtype=np.int32).reshape(2, 4)
    graph_results = run_node(node, [data])
    assert all_arrays_equal(graph_results, expected_output)


def test_split_2d_splits_input():
    data = np.arange(8, dtype=np.int32).reshape(2, 4)
    splits = np.array([3, 1]).astype(np.int64)
    node = onnx.helper.make_node(
        "Split", inputs=["x", "splits"], outputs=["a", "b"], axis=1,
    )
    expected_outputs = [
        np.array([[0, 1, 2], [4, 5, 6]], dtype=np.int32),
        np.array([[3], [7]], dtype=np.int32),
    ]
    graph_results = run_node(node, [data, splits])
    assert all_arrays_equal(graph_results, expected_outputs)


def test_split_1d():
    # 1D
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32)

    node = onnx.helper.make_node("Split", inputs=["input"], outputs=["z", "w"], axis=0)
    expected_outputs = [
        np.array([1.0, 2.0, 3.0]).astype(np.float32),
        np.array([4.0, 5.0, 6.0]).astype(np.float32),
    ]
    graph_results = run_node(node, [data])
    assert all_arrays_equal(graph_results, expected_outputs)

    splits = np.array([2, 3, 1]).astype(np.int64)
    node = onnx.helper.make_node(
        "Split", inputs=["input", "splits"], outputs=["y", "z", "w"], axis=0,
    )
    expected_outputs = [
        np.array([1.0, 2.0]).astype(np.float32),
        np.array([3.0, 4.0, 5.0]).astype(np.float32),
        np.array([6.0]).astype(np.float32),
    ]
    graph_results = run_node(node, [data, splits])
    assert all_arrays_equal(graph_results, expected_outputs)

    # Default values
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32)

    node = onnx.helper.make_node("Split", inputs=["input"], outputs=["y", "z", "w"])
    expected_outputs = [
        np.array([1.0, 2.0]).astype(np.float32),
        np.array([3.0, 4.0]).astype(np.float32),
        np.array([5.0, 6.0]).astype(np.float32),
    ]
    graph_results = run_node(node, [data])
    assert all_arrays_equal(graph_results, expected_outputs)

    splits = np.array([2, 4]).astype(np.int64)
    node = onnx.helper.make_node(
        "Split", inputs=["input", "splits"], outputs=["y", "z"],
    )
    expected_outputs = [
        np.array([1.0, 2.0]).astype(np.float32),
        np.array([3.0, 4.0, 5.0, 6.0]).astype(np.float32),
    ]
    graph_results = run_node(node, [data, splits])
    assert all_arrays_equal(graph_results, expected_outputs)


def test_depth_to_space():
    b, c, h, w = shape = (2, 8, 3, 3)  # noqa: VNE001
    blocksize = 2
    data = np.random.random_sample(shape).astype(np.float32)
    tmp = np.reshape(data, [b, blocksize, blocksize, c // (blocksize ** 2), h, w])
    tmp = np.transpose(tmp, [0, 3, 4, 1, 5, 2])
    expected_output = np.reshape(
        tmp, [b, c // (blocksize ** 2), h * blocksize, w * blocksize],
    )

    node = onnx.helper.make_node("DepthToSpace", inputs=["x"], outputs=["y"], blocksize=blocksize)
    graph_results = run_node(node, [data])
    assert np.array_equal(graph_results, [expected_output])

    # (1, 4, 2, 3) input tensor
    data = np.array(
        [
            [
                [[0, 1, 2], [3, 4, 5]],
                [[6, 7, 8], [9, 10, 11]],
                [[12, 13, 14], [15, 16, 17]],
                [[18, 19, 20], [21, 22, 23]],
            ],
        ],
    ).astype(np.float32)
    # (1, 1, 4, 6) output tensor
    expected_output = np.array(
        [
            [
                [
                    [0, 6, 1, 7, 2, 8],
                    [12, 18, 13, 19, 14, 20],
                    [3, 9, 4, 10, 5, 11],
                    [15, 21, 16, 22, 17, 23],
                ],
            ],
        ],
    ).astype(np.float32)

    graph_results = run_node(node, [data])
    assert np.array_equal(graph_results, [expected_output])
