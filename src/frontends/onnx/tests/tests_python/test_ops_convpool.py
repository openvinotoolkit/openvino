# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx
import pytest
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.onnx_cpp2py_export.checker import ValidationError

from tests.runtime import get_runtime
from tests.tests_python.utils import get_node_model, import_onnx_model, run_model, run_node


@pytest.fixture()
def ndarray_1x1x4x4():
    return np.array(
        [[11, 12, 13, 14], [15, 16, 17, 18], [19, 20, 21, 22], [23, 24, 25, 26]], dtype=np.float32,
    ).reshape([1, 1, 4, 4])


def make_onnx_model_for_conv_op(x_shape, weights_shape, transpose=False, **attributes):
    output_shape = ()  # We don't need output shape to be accurate for these tests

    if transpose:
        node_op = "ConvTranspose"
    else:
        node_op = "Conv"

    node = make_node(node_op, ["X", "weight"], ["Y"], name="test_node", **attributes)
    graph = make_graph(
        [node],
        "test_graph",
        [
            make_tensor_value_info("X", onnx.TensorProto.FLOAT, x_shape),
            make_tensor_value_info("weight", onnx.TensorProto.FLOAT, weights_shape),
        ],
        [make_tensor_value_info("Y", onnx.TensorProto.FLOAT, output_shape)],
    )
    model = make_model(graph, producer_name="OpenVINO ONNX Frontend")
    return model


def import_and_compute_conv(inputs, weights, transpose=False, **attributes):
    inputs, weights = np.array(inputs), np.array(weights)
    onnx_model = make_onnx_model_for_conv_op(inputs.shape, weights.shape, transpose=transpose, **attributes)
    model = import_onnx_model(onnx_model)
    computation = get_runtime().computation(model)
    return computation(inputs, weights)[0]


def test_2d_conv():
    # x should have shape N(batch) x C x H x W
    input_x = np.array(
        [
            [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    ).reshape(1, 1, 9, 9)

    # filter weights should have shape M x C x kH x kW
    input_filter = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=np.float32).reshape(
        [1, 1, 3, 3],
    )

    # convolution with padding=1 should produce 9 x 9 output:
    result = import_and_compute_conv(input_x, input_filter, pads=(1, 1, 1, 1), strides=(1, 1))
    assert np.array_equal(
        result,
        np.array(
            [
                [
                    [
                        [0.0, -15.0, -15.0, 15.0, 15.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, -20.0, -20.0, 20.0, 20.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, -20.0, -20.0, 20.0, 20.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, -20.0, -20.0, 20.0, 20.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, -20.0, -20.0, 20.0, 20.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, -20.0, -20.0, 20.0, 20.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, -20.0, -20.0, 20.0, 20.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, -20.0, -20.0, 20.0, 20.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, -15.0, -15.0, 15.0, 15.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                ],
            ],
            dtype=np.float32,
        ),
    )

    # convolution with padding=0 should produce 7 x 7 output:
    result = import_and_compute_conv(input_x, input_filter, pads=(0, 0, 0, 0), strides=(1, 1))
    assert np.array_equal(
        result,
        np.array(
            [
                [
                    [
                        [-20, -20, 20, 20, 0, 0, 0],
                        [-20, -20, 20, 20, 0, 0, 0],
                        [-20, -20, 20, 20, 0, 0, 0],
                        [-20, -20, 20, 20, 0, 0, 0],
                        [-20, -20, 20, 20, 0, 0, 0],
                        [-20, -20, 20, 20, 0, 0, 0],
                        [-20, -20, 20, 20, 0, 0, 0],
                    ],
                ],
            ],
            dtype=np.float32,
        ),
    )

    # convolution with strides=2 should produce 4 x 4 output:
    result = import_and_compute_conv(input_x, input_filter, pads=(0, 0, 0, 0), strides=(2, 2))
    assert np.array_equal(
        result,
        np.array(
            [
                [
                    [
                        [-20.0, 20.0, 0.0, 0.0],
                        [-20.0, 20.0, 0.0, 0.0],
                        [-20.0, 20.0, 0.0, 0.0],
                        [-20.0, 20.0, 0.0, 0.0],
                    ],
                ],
            ],
            dtype=np.float32,
        ),
    )

    # convolution with dilations=2 should produce 5 x 5 output:
    result = import_and_compute_conv(input_x, input_filter, dilations=(2, 2))
    assert np.array_equal(
        result,
        np.array(
            [
                [
                    [
                        [0, 0, 20, 20, 0],
                        [0, 0, 20, 20, 0],
                        [0, 0, 20, 20, 0],
                        [0, 0, 20, 20, 0],
                        [0, 0, 20, 20, 0],
                    ],
                ],
            ],
            dtype=np.float32,
        ),
    )


def test_3d_conv():
    # x should have shape N(batch) x C x H x W x D
    input_x = np.array(
        [
            [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    ).reshape([1, 1, 9, 9, 1])
    input_x = np.broadcast_to(input_x, (1, 1, 9, 9, 4))

    # filter weights should have shape M x C x kH x kW x kD
    input_filter = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=np.float32).reshape(
        [1, 1, 3, 3, 1],
    )
    input_filter = np.broadcast_to(input_filter, (1, 1, 3, 3, 3))

    # convolution with padding=0 should produce 7 x 7 x 2 output:
    result = import_and_compute_conv(
        input_x, input_filter, dilations=(1, 1, 1), pads=(0, 0, 0, 0, 0, 0), strides=(1, 1, 1),
    )

    assert np.array_equal(
        np.moveaxis(result.squeeze(), (0, 1, 2), (1, 2, 0)),
        np.array(
            [
                [
                    [-60.0, -60.0, 60.0, 60.0, 0.0, 0.0, 0.0],
                    [-60.0, -60.0, 60.0, 60.0, 0.0, 0.0, 0.0],
                    [-60.0, -60.0, 60.0, 60.0, 0.0, 0.0, 0.0],
                    [-60.0, -60.0, 60.0, 60.0, 0.0, 0.0, 0.0],
                    [-60.0, -60.0, 60.0, 60.0, 0.0, 0.0, 0.0],
                    [-60.0, -60.0, 60.0, 60.0, 0.0, 0.0, 0.0],
                    [-60.0, -60.0, 60.0, 60.0, 0.0, 0.0, 0.0],
                ],
                [
                    [-60.0, -60.0, 60.0, 60.0, 0.0, 0.0, 0.0],
                    [-60.0, -60.0, 60.0, 60.0, 0.0, 0.0, 0.0],
                    [-60.0, -60.0, 60.0, 60.0, 0.0, 0.0, 0.0],
                    [-60.0, -60.0, 60.0, 60.0, 0.0, 0.0, 0.0],
                    [-60.0, -60.0, 60.0, 60.0, 0.0, 0.0, 0.0],
                    [-60.0, -60.0, 60.0, 60.0, 0.0, 0.0, 0.0],
                    [-60.0, -60.0, 60.0, 60.0, 0.0, 0.0, 0.0],
                ],
            ],
            dtype=np.float32,
        ),
    )


def test_2d_conv_transpose():
    # x should have shape N(batch) x C x H x W
    input_x = np.array(
        [
            [0.0, -15.0, -15.0, 15.0, 15.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -20.0, -20.0, 20.0, 20.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -20.0, -20.0, 20.0, 20.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -20.0, -20.0, 20.0, 20.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -20.0, -20.0, 20.0, 20.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -20.0, -20.0, 20.0, 20.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -20.0, -20.0, 20.0, 20.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -20.0, -20.0, 20.0, 20.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -15.0, -15.0, 15.0, 15.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    ).reshape([1, 1, 9, 9])

    # filter weights should have shape M x C x kH x kW
    input_filter = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=np.float32).reshape(
        [1, 1, 3, 3],
    )

    # deconvolution with padding=1 should produce 9 x 9 output:
    result = import_and_compute_conv(input_x, input_filter, transpose=True, pads=(1, 1, 1, 1), strides=(1, 1))

    assert np.array_equal(
        result.reshape([9, 9]),
        np.array(
            [
                [-50.0, -50.0, 100.0, 100.0, -50.0, -50.0, 0.0, 0.0, 0.0],
                [-75.0, -75.0, 150.0, 150.0, -75.0, -75.0, 0.0, 0.0, 0.0],
                [-80.0, -80.0, 160.0, 160.0, -80.0, -80.0, 0.0, 0.0, 0.0],
                [-80.0, -80.0, 160.0, 160.0, -80.0, -80.0, 0.0, 0.0, 0.0],
                [-80.0, -80.0, 160.0, 160.0, -80.0, -80.0, 0.0, 0.0, 0.0],
                [-80.0, -80.0, 160.0, 160.0, -80.0, -80.0, 0.0, 0.0, 0.0],
                [-80.0, -80.0, 160.0, 160.0, -80.0, -80.0, 0.0, 0.0, 0.0],
                [-75.0, -75.0, 150.0, 150.0, -75.0, -75.0, 0.0, 0.0, 0.0],
                [-50.0, -50.0, 100.0, 100.0, -50.0, -50.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_pad_opset_1():
    inputs = np.ones((2, 2), dtype=np.float32)
    outputs = np.pad(inputs, pad_width=1, mode="constant")

    model = get_node_model("Pad", inputs, paddings=[1, 1, 1, 1])
    graph_results = run_model(model, [inputs])
    assert np.array_equal(graph_results, [outputs])

    inputs = np.random.randn(1, 3, 4, 5).astype(np.float32)
    outputs = np.pad(inputs, pad_width=((0, 0), (0, 0), (1, 2), (3, 4)), mode="constant")

    model = get_node_model("Pad", inputs, mode="constant", paddings=[0, 0, 1, 3, 0, 0, 2, 4])
    graph_results = run_model(model, [inputs])
    assert np.array_equal(graph_results, [outputs])

    # incorrect paddings rank
    inputs = np.ones((2, 2), dtype=np.float32)
    model = get_node_model("Pad", inputs, paddings=[0, 1, 1, 3, 1, 2])
    with pytest.raises(RuntimeError):
        run_model(model, [inputs])

    # no paddings arttribute
    model = get_node_model("Pad", inputs)
    with pytest.raises(ValidationError):
        import_onnx_model(model)


def test_pad_opset_2():
    inputs = np.ones((2, 2), dtype=np.float32)
    outputs = np.pad(inputs, pad_width=1, mode="constant")

    model = get_node_model("Pad", inputs, opset=2, pads=[1, 1, 1, 1])
    graph_results = run_model(model, [inputs])
    assert np.array_equal(graph_results, [outputs])

    inputs = np.random.randn(1, 3, 4, 5).astype(np.float32)
    outputs = np.pad(inputs, pad_width=((0, 0), (0, 0), (1, 2), (3, 4)), mode="constant")

    model = get_node_model("Pad", inputs, opset=2, mode="constant", pads=[0, 0, 1, 3, 0, 0, 2, 4])
    graph_results = run_model(model, [inputs])
    assert np.array_equal(graph_results, [outputs])

    # incorrect pads rank
    inputs = np.ones((2, 2), dtype=np.float32)
    model = get_node_model("Pad", inputs, opset=2, pads=[0, 1, 1, 3, 1, 2])
    with pytest.raises(RuntimeError):
        run_model(model, [inputs])


def test_pad_negative_values_begin():
    inputs = np.ones((2, 2), dtype=np.float32)

    # Axis 1 begin
    model = get_node_model("Pad", inputs, opset=2, pads=[-1, 0, 0, 0])
    graph_result = run_model(model, [inputs])[0]
    assert np.array_equal(graph_result, np.array([[1, 1]]))

    # Axis 2 begin
    model = get_node_model("Pad", inputs, opset=2, pads=[0, -1, 0, 0])
    graph_result = run_model(model, [inputs])[0]
    assert np.array_equal(graph_result, np.array([[1], [1]]))


def test_pad_negative_values_end():
    inputs = np.ones((2, 2), dtype=np.float32)

    # Axis 1 end
    model = get_node_model("Pad", inputs, opset=2, pads=[0, 0, -1, 0])
    graph_result = run_model(model, [inputs])[0]
    assert np.array_equal(graph_result, np.array([[1.0, 1.0]]))

    # Axis 2 end
    model = get_node_model("Pad", inputs, opset=2, pads=[0, 0, 0, -1])
    graph_result = run_model(model, [inputs])[0]
    assert np.array_equal(graph_result, np.array([[1], [1]]))


def test_pool_average(ndarray_1x1x4x4):
    inputs = ndarray_1x1x4x4
    node = onnx.helper.make_node("AveragePool", inputs=["x"], outputs=["y"], kernel_shape=(2, 2), strides=(2, 2))
    outputs = np.array([[13.5, 15.5], [21.5, 23.5]], dtype=np.float32).reshape([1, 1, 2, 2])
    graph_results = run_node(node, [inputs])
    assert np.array_equal(graph_results, [outputs])

    node = onnx.helper.make_node(
        "AveragePool", inputs=["x"], outputs=["y"], kernel_shape=(2, 2), strides=(2, 2), pads=(1, 1, 1, 1),
    )
    outputs = np.array([[11, 12.5, 14], [17, 18.5, 20], [23, 24.5, 26]], dtype=np.float32).reshape([1, 1, 3, 3])
    graph_results = run_node(node, [inputs])
    assert np.array_equal(graph_results, [outputs])


def test_pool_average_3d(ndarray_1x1x4x4):
    inputs = np.broadcast_to(ndarray_1x1x4x4, (1, 1, 4, 4, 4))
    node = onnx.helper.make_node("AveragePool", inputs=["x"], outputs=["y"], kernel_shape=(2, 2, 2), strides=(2, 2, 2))
    outputs = np.array([[[13.5, 15.5], [21.5, 23.5]], [[13.5, 15.5], [21.5, 23.5]]], dtype=np.float32).reshape(
        [1, 1, 2, 2, 2],
    )
    graph_results = run_node(node, [inputs])
    assert np.array_equal(graph_results, [outputs])


def test_pool_max(ndarray_1x1x4x4):
    node = onnx.helper.make_node("MaxPool", inputs=["x"], outputs=["y"], kernel_shape=(2, 2), strides=(2, 2))

    inputs = ndarray_1x1x4x4
    outputs = np.array([[16, 18], [24, 26]], dtype=np.float32).reshape([1, 1, 2, 2])

    graph_results = run_node(node, [inputs], opset_version=7)
    assert np.array_equal(graph_results, [outputs])


def test_pool_global_max(ndarray_1x1x4x4):
    node = onnx.helper.make_node("GlobalMaxPool", inputs=["x"], outputs=["y"])

    inputs = ndarray_1x1x4x4
    outputs = np.array([26], dtype=np.float32).reshape([1, 1, 1, 1])

    graph_results = run_node(node, [inputs])
    assert np.array_equal(graph_results, [outputs])


def test_pool_global_average(ndarray_1x1x4x4):
    node = onnx.helper.make_node("GlobalAveragePool", inputs=["x"], outputs=["y"])

    inputs = ndarray_1x1x4x4
    outputs = np.array([18.5], dtype=np.float32).reshape([1, 1, 1, 1])

    graph_results = run_node(node, [inputs])
    assert np.array_equal(graph_results, [outputs])


def test_pool_global_average_3d(ndarray_1x1x4x4):
    inputs = np.broadcast_to(ndarray_1x1x4x4, (1, 1, 4, 4, 4))

    node = onnx.helper.make_node("GlobalAveragePool", inputs=["x"], outputs=["y"])
    outputs = np.array([18.5], dtype=np.float32).reshape([1, 1, 1, 1, 1])
    graph_results = run_node(node, [inputs])
    assert np.array_equal(graph_results, [outputs])
