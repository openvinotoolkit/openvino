# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx
import onnx.mapping
import pytest
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

from openvino.runtime.exceptions import OVTypeError
from tests.runtime import get_runtime
from tests.tests_python.utils import get_node_model, import_onnx_model, run_model, run_node


@pytest.mark.parametrize(
    "input_data",
    [
        np.array([-4, 0, 5, -10], dtype=np.float32),
        np.array([[-4, 0, 5, -10], [-4, 0, 5, -10]], dtype=np.float32),
        np.array([[[1, 2], [-3, 4]], [[1, -2], [3, 4]]], dtype=np.float32),
    ],
)
def test_abs(input_data):
    expected_output = np.abs(input_data)
    node = onnx.helper.make_node("Abs", inputs=["x"], outputs=["y"])
    graph_results = run_node(node, [input_data])
    assert np.array_equal(graph_results, [expected_output])


@pytest.mark.parametrize(
    "input_data",
    [
        np.array([4, 0, 5, 10]),
        np.array([[4, 0, 5, 10], [4, 0, 5, 10]]),
        np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]]),
    ],
)
def test_sqrt(input_data):
    input_data = input_data.astype(np.float32)
    expected_output = np.sqrt(input_data)
    node = onnx.helper.make_node("Sqrt", inputs=["x"], outputs=["y"])
    graph_results = run_node(node, [input_data])
    assert np.allclose(graph_results, [expected_output])


@pytest.mark.parametrize(
    "input_data",
    [
        np.array([4, 0, 5, 10]),
        np.array([[4, 0, 5, 10], [4, 0, 5, 10]]),
        np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]]),
    ],
)
def test_exp(input_data):
    input_data = input_data.astype(np.float32)
    expected_output = np.exp(input_data)
    node = onnx.helper.make_node("Exp", inputs=["x"], outputs=["y"])
    graph_results = run_node(node, [input_data])
    assert np.allclose(graph_results, [expected_output])


@pytest.mark.parametrize(
    "input_data",
    [
        np.array([4, 2, 5, 10]),
        np.array([[4, 1, 5, 10], [4, 2, 5, 10]]),
        np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]]),
    ],
)
def test_log(input_data):
    input_data = input_data.astype(np.float32)
    expected_output = np.log(input_data)
    node = onnx.helper.make_node("Log", inputs=["x"], outputs=["y"])
    graph_results = run_node(node, [input_data])
    assert np.allclose(graph_results, [expected_output])


@pytest.mark.parametrize(
    "input_data",
    [
        np.array([-4, 0, 5, -10], dtype=np.float32),
        np.array([[-4, 0, 5, -10], [-4, 0, 5, -10]], dtype=np.float32),
        np.array([[[1, 2], [-3, 4]], [[1, -2], [3, 4]]], dtype=np.float32),
    ],
)
def test_neg(input_data):
    expected_output = np.negative(input_data)
    node = onnx.helper.make_node("Neg", inputs=["x"], outputs=["y"])
    graph_results = run_node(node, [input_data])
    assert np.array_equal(graph_results, [expected_output])


@pytest.mark.parametrize(
    "input_data",
    [
        np.array([-4.2, 0.43, 5.99, -10.01]),
        np.array([[-4.5, 0.99, 5.01, -10.00], [-4.5, 0.5, 5.1, 10.01]]),
        np.array([[[1, 2], [-3, 4]], [[1, -2], [3, 4]]]) / 6,
    ],
)
def test_floor(input_data):
    input_data = input_data.astype(np.float32)
    expected_output = np.floor(input_data)
    node = onnx.helper.make_node("Floor", inputs=["x"], outputs=["y"])
    graph_results = run_node(node, [input_data])
    assert np.array_equal(graph_results, [expected_output])


@pytest.mark.parametrize(
    "input_data",
    [
        np.array([-4.2, 0, 5.99, -10.01]),
        np.array([[-4.5, 0.99, 5.01, -10.00], [-4.5, 0.5, 5.1, 10.01]]),
        np.array([[[1, 2], [-3, 4]], [[1, -2], [3, 4]]]) / 6,
    ],
)
def test_ceil(input_data):
    input_data = input_data.astype(np.float32)
    expected_output = np.ceil(input_data)
    node = onnx.helper.make_node("Ceil", inputs=["x"], outputs=["y"])
    graph_results = run_node(node, [input_data])
    assert np.array_equal(graph_results, [expected_output])


@pytest.mark.parametrize(
    ("min_value", "max_value"),
    [(np.finfo(np.float32).min, np.finfo(np.float32).max), (-0.5, 0.5), (0.0, np.finfo(np.float32).max)],
)
def test_clip(min_value, max_value):
    np.random.seed(133391)
    input_data = np.float32(-100.0) + np.random.randn(3, 4, 5).astype(np.float32) * np.float32(200.0)
    model = get_node_model("Clip", input_data, opset=10, min=float(min_value), max=float(max_value))
    result = run_model(model, [input_data])
    expected = np.clip(input_data, min_value, max_value)
    assert np.allclose(result, [expected])


def test_clip_default():
    np.random.seed(133391)
    input_data = -100.0 + np.random.randn(3, 4, 5).astype(np.float32) * 200.0

    model = get_node_model("Clip", input_data, opset=10, min=0.0)
    result = run_model(model, [input_data])
    expected = np.clip(input_data, np.float32(0.0), np.finfo(np.float32).max)
    assert np.allclose(result, [expected])

    model = get_node_model("Clip", input_data, opset=10, max=0.0)
    result = run_model(model, [input_data])
    expected = np.clip(input_data, np.finfo(np.float32).min, np.float32(0.0))
    assert np.allclose(result, [expected])


@pytest.mark.parametrize(
    "input_data",
    [
        np.array([-4.2, 1, 5.99, -10.01]),
        np.array([[-4.5, 0.99, 5.01, -10.00], [-4.5, 0.5, 5.1, 10.01]]),
        np.array([[[1, 2], [-3, 4]], [[1, -2], [3, 4]]]) / 6,
    ],
)
def test_reciprocal(input_data):
    input_data = input_data.astype(np.float32)
    expected_output = np.reciprocal(input_data)
    node = onnx.helper.make_node("Reciprocal", inputs=["x"], outputs=["y"])
    graph_results = run_node(node, [input_data])
    assert np.allclose(graph_results, [expected_output])


@pytest.mark.parametrize(("axis", "dim1", "dim2"), [(0, 1, 60), (1, 3, 20), (2, 12, 5)])
def test_hardmax(axis, dim1, dim2):
    def hardmax_2d(data):
        return np.eye(data.shape[1], dtype=data.dtype)[np.argmax(data, axis=1)]

    np.random.seed(133391)
    data = np.random.rand(3, 4, 5).astype(np.float32)
    expected = hardmax_2d(data.reshape(dim1, dim2)).reshape(3, 4, 5)
    node = onnx.helper.make_node("Hardmax", inputs=["x"], outputs=["y"], axis=axis)
    graph_results = run_node(node, [data], opset_version=12)
    assert np.allclose(graph_results, [expected])


def test_hardmax_special_cases():
    def hardmax_2d(data):
        return np.eye(data.shape[1], dtype=data.dtype)[np.argmax(data, axis=1)]

    np.random.seed(133391)
    data = np.random.rand(3, 4, 5).astype(np.float32)

    # default axis=1
    expected = hardmax_2d(data.reshape(3, 20)).reshape(3, 4, 5)
    node = onnx.helper.make_node("Hardmax", inputs=["x"], outputs=["y"])
    graph_results = run_node(node, [data], opset_version=12)
    assert np.allclose(graph_results, [expected])

    expected = hardmax_2d(data.reshape(12, 5)).reshape(3, 4, 5)
    node = onnx.helper.make_node("Hardmax", inputs=["x"], outputs=["y"], axis=-1)
    graph_results = run_node(node, [data], opset_version=12)
    assert np.allclose(graph_results, [expected])

    node = onnx.helper.make_node("Hardmax", inputs=["x"], outputs=["y"], axis=3)
    with pytest.raises(RuntimeError):
        graph_results = run_node(node, [data], opset_version=12)

    # For multiple occurrences of the maximal values, the first occurrence is selected
    # for one-hot output
    data = np.array([[3, 3, 3, 1]]).astype(np.float32)
    expected = np.array([[1, 0, 0, 0]]).astype(np.float32)
    node = onnx.helper.make_node("Hardmax", inputs=["x"], outputs=["y"])
    graph_results = run_node(node, [data], opset_version=12)
    assert np.allclose(graph_results, [expected])


def test_hardsigmoid():
    def hardsigmoid(data, alpha=0.2, beta=0.5):
        return np.clip(alpha * data + beta, 0, 1)

    np.random.seed(133391)
    alpha = np.random.rand()
    beta = np.random.rand()
    data = np.random.rand(3, 4, 5).astype(np.float32)

    expected = hardsigmoid(data, alpha, beta)
    node = onnx.helper.make_node("HardSigmoid", inputs=["x"], outputs=["y"], alpha=alpha, beta=beta)
    graph_results = run_node(node, [data])
    assert np.allclose(graph_results, [expected])

    expected = hardsigmoid(data)
    node = onnx.helper.make_node("HardSigmoid", inputs=["x"], outputs=["y"])
    graph_results = run_node(node, [data])
    assert np.allclose(graph_results, [expected])


def test_logsoftmax():
    def logsoftmax_2d(value):
        max_x = np.max(value, axis=1).reshape((-1, 1))
        exp_x = np.exp(value - max_x)
        return value - max_x - np.log(np.sum(exp_x, axis=1).reshape((-1, 1)))

    np.random.seed(133391)
    data = np.random.randn(3, 4, 5).astype(np.float32)

    node = onnx.helper.make_node("LogSoftmax", inputs=["x"], outputs=["y"], axis=0)
    expected = logsoftmax_2d(data.reshape(1, 60)).reshape(3, 4, 5)
    graph_results = run_node(node, [data], opset_version=12)
    assert np.allclose(graph_results, [expected])

    node = onnx.helper.make_node("LogSoftmax", inputs=["x"], outputs=["y"], axis=1)
    expected = logsoftmax_2d(data.reshape(3, 20)).reshape(3, 4, 5)
    graph_results = run_node(node, [data], opset_version=12)
    assert np.allclose(graph_results, [expected])

    # default axis is 1
    node = onnx.helper.make_node("LogSoftmax", inputs=["x"], outputs=["y"])
    graph_results = run_node(node, [data], opset_version=12)
    assert np.allclose(graph_results, [expected])

    node = onnx.helper.make_node("LogSoftmax", inputs=["x"], outputs=["y"], axis=2)
    expected = logsoftmax_2d(data.reshape(12, 5)).reshape(3, 4, 5)
    graph_results = run_node(node, [data], opset_version=12)
    assert np.allclose(graph_results, [expected])

    node = onnx.helper.make_node("LogSoftmax", inputs=["x"], outputs=["y"], axis=3)
    with pytest.raises(RuntimeError):
        graph_results = run_node(node, [data], opset_version=12)


def test_softplus():
    def softplus(value):
        return np.where(value < 20, np.log(np.exp(value) + 1), value)

    np.random.seed(133391)
    data = np.random.randn(3, 4, 5).astype(np.float32)

    node = onnx.helper.make_node("Softplus", inputs=["x"], outputs=["y"])
    expected = softplus(data)
    graph_results = run_node(node, [data])
    assert np.allclose(graph_results, [expected])


def test_softsign():
    def softsign(value):
        return value / (1 + np.abs(value))

    np.random.seed(133391)
    data = np.random.randn(3, 4, 5).astype(np.float32)

    node = onnx.helper.make_node("Softsign", inputs=["x"], outputs=["y"])
    expected = softsign(data)
    graph_results = run_node(node, [data])
    assert np.allclose(graph_results, [expected])


def test_identity():
    np.random.seed(133391)
    shape = [2, 4]
    input_data = np.random.randn(*shape).astype(np.float32)

    identity_node = make_node("Identity", inputs=["x"], outputs=["y"])
    graph_results = run_node(identity_node, [input_data])
    assert np.array_equal(graph_results, [input_data])

    node1 = make_node("Add", inputs=["A", "B"], outputs=["add1"], name="add_node1")
    node2 = make_node("Identity", inputs=["add1"], outputs=["identity1"], name="identity_node1")
    node3 = make_node("Abs", inputs=["identity1"], outputs=["Y"], name="abs_node1")

    graph = make_graph(
        [node1, node2, node3],
        "test_graph",
        [
            make_tensor_value_info("A", onnx.TensorProto.FLOAT, shape),
            make_tensor_value_info("B", onnx.TensorProto.FLOAT, shape),
        ],
        [make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape)],
    )
    model = make_model(graph, producer_name="OpenVINO ONNX Frontend")
    graph_model = import_onnx_model(model)
    runtime = get_runtime()
    computation = runtime.computation(graph_model)
    graph_results = computation(input_data, input_data)
    expected_result = np.abs(input_data + input_data)

    assert np.array_equal(graph_results[0], expected_result)


@pytest.mark.parametrize(("val_type", "input_data"), [(np.dtype(bool), np.zeros((2, 2), dtype=int))])
def test_cast_to_bool(val_type, input_data):
    expected = np.array(input_data, dtype=val_type)

    model = get_node_model("Cast", input_data, opset=6, to=onnx.helper.np_dtype_to_tensor_dtype(val_type))
    result = run_model(model, [input_data])
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    ("val_type", "range_start", "range_end", "in_dtype"),
    [
        (np.dtype(np.float32), -8, 8, np.dtype(np.int32)),
        (np.dtype(np.float64), -16383, 16383, np.dtype(np.int64)),
    ],
)
def test_cast_to_float(val_type, range_start, range_end, in_dtype):
    np.random.seed(133391)
    input_data = np.random.randint(range_start, range_end, size=(2, 2), dtype=in_dtype)
    expected = np.array(input_data, dtype=val_type)

    model = get_node_model("Cast", input_data, opset=6, to=onnx.helper.np_dtype_to_tensor_dtype(val_type))
    result = run_model(model, [input_data])
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "val_type", [np.dtype(np.int8),
                 np.dtype(np.int16),
                 np.dtype(np.int32),
                 np.dtype(np.int64)],
)
def test_cast_to_int(val_type):
    np.random.seed(133391)
    random_data = np.random.rand(2, 3, 4) * 16
    input_data = np.ceil(-8 + random_data).astype(val_type)
    expected = np.array(input_data, dtype=val_type)

    model = get_node_model("Cast", input_data, opset=6, to=onnx.helper.np_dtype_to_tensor_dtype(val_type))
    result = run_model(model, [input_data])
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "val_type", [np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.uint32), np.dtype(np.uint64)],
)
def test_cast_to_uint(val_type):
    np.random.seed(133391)
    input_data = np.ceil(np.random.rand(2, 3, 4) * 16).astype(val_type)
    expected = np.array(input_data, dtype=val_type)

    model = get_node_model("Cast", input_data, opset=6, to=onnx.helper.np_dtype_to_tensor_dtype(val_type))
    result = run_model(model, [input_data])
    assert np.allclose(result, expected)


def test_cast_errors():
    from onnx.onnx_cpp2py_export.checker import ValidationError

    np.random.seed(133391)
    input_data = np.ceil(np.random.rand(2, 3, 4) * 16)

    # missing 'to' attribute
    node = onnx.helper.make_node("Cast", inputs=["A"], outputs=["B"])
    input_tensors = [
        make_tensor_value_info(name, onnx.TensorProto.FLOAT, value.shape)
        for name, value in zip(node.input, [input_data])
    ]
    output_tensors = [make_tensor_value_info(node.output[0], onnx.TensorProto.FLOAT16, input_data.shape)]  # type: ignore

    graph = make_graph([node], "compute_graph", input_tensors, output_tensors)
    model = make_model(graph, producer_name="OpenVINO ONNX Frontend")
    with pytest.raises(ValidationError):
        import_onnx_model(model)

    # unsupported data type representation
    node = onnx.helper.make_node("Cast", inputs=["A"], outputs=["B"], to=1.2345)
    input_tensors = [
        make_tensor_value_info(name, onnx.TensorProto.FLOAT, value.shape)
        for name, value in zip(node.input, [input_data])
    ]
    output_tensors = [make_tensor_value_info(node.output[0], onnx.TensorProto.INT32, input_data.shape)]  # type: ignore

    graph = make_graph([node], "compute_graph", input_tensors, output_tensors)
    model = make_model(graph, producer_name="OpenVINO ONNX Frontend")
    with pytest.raises(ValidationError):
        import_onnx_model(model)

    # unsupported input tensor data type:
    node = onnx.helper.make_node("Cast", inputs=["A"], outputs=["B"], to=onnx.TensorProto.INT32)
    input_tensors = [
        make_tensor_value_info(name, onnx.TensorProto.COMPLEX64, value.shape)
        for name, value in zip(node.input, [input_data])
    ]
    output_tensors = [make_tensor_value_info(node.output[0], onnx.TensorProto.INT32, input_data.shape)]  # type: ignore

    graph = make_graph([node], "compute_graph", input_tensors, output_tensors)
    model = make_model(graph, producer_name="OpenVINO ONNX Frontend")
    with pytest.raises((RuntimeError, OVTypeError)):
        import_onnx_model(model)

    # unsupported output tensor data type:
    node = onnx.helper.make_node("Cast", inputs=["A"], outputs=["B"], to=onnx.TensorProto.COMPLEX128)
    input_tensors = [
        make_tensor_value_info(name, onnx.TensorProto.FLOAT, value.shape)
        for name, value in zip(node.input, [input_data])
    ]
    output_tensors = [make_tensor_value_info(node.output[0], onnx.TensorProto.COMPLEX128, input_data.shape)]  # type: ignore

    graph = make_graph([node], "compute_graph", input_tensors, output_tensors)
    model = make_model(graph, producer_name="OpenVINO ONNX Frontend")
    with pytest.raises(RuntimeError):
        import_onnx_model(model)


@pytest.mark.parametrize("value_type",
                         [pytest.param(np.float64),
                          pytest.param(np.float32)])
def test_constant(value_type):
    values = np.random.randn(5, 5).astype(value_type)
    node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["values"],
        value=onnx.helper.make_tensor(
            name="const_tensor",
            data_type=onnx.helper.np_dtype_to_tensor_dtype(np.dtype(value_type)),
            dims=values.shape,
            vals=values.flatten(),
        ),
    )

    graph_results = run_node(node, [])
    assert np.allclose(graph_results, [values])


def test_constant_err():
    values = np.random.randn(5, 5).astype(np.float16)
    node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["values"],
        value=onnx.helper.make_tensor(
            name="const_tensor",
            data_type=onnx.helper.np_dtype_to_tensor_dtype(np.dtype(np.float16)),
            dims=values.shape,
            vals=values.flatten(),
        ),
    )

    graph_results = run_node(node, [])
    assert np.allclose(graph_results, [values])


@pytest.mark.parametrize(
    ("shape", "shift"),
    [
        ((1, 1), -1),
        ((2, 4), 5),
        ((2, 4), 15),
        ((2, 4), -5),
        ((2, 4), -15),
        ((4, 4), 0),
        ((4, 4), 1),
        ((4, 4), -1),
        ((4, 4), 2),
        ((4, 4), -2),
        ((4, 4), 3),
        ((4, 4), -3),
        ((3, 4), 0),
        ((3, 4), 1),
        ((3, 4), -1),
        ((3, 4), 2),
        ((3, 4), -2),
        ((5, 3), 0),
        ((5, 3), 1),
        ((5, 3), -1),
        ((5, 3), 2),
        ((5, 3), -2),
    ],
)
def test_eye_like(shape, shift):
    input_tensor = np.arange(np.prod(shape)).reshape(shape)

    node = onnx.helper.make_node("EyeLike", inputs=["x"], outputs=["y"], k=shift)
    result = run_node(node, [input_tensor])[0]

    assert np.allclose(result, np.eye(shape[0], shape[1], k=shift))
