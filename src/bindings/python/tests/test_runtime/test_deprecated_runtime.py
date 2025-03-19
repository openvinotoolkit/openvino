# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from pathlib import Path
from contextlib import nullcontext as does_not_raise
import warnings
import operator

with pytest.warns(DeprecationWarning, match="The `openvino.runtime` module is deprecated and will be removed in the 2026.0 release."):
    import openvino.runtime as ov

import openvino.runtime.opset13 as ops
from openvino.runtime import (
    Model,
    Core,
    AsyncInferQueue,
    Strides,
    Shape,
    PartialShape,
    serialize,
    Type,
)
import openvino.runtime.opset8 as ops8
from openvino.runtime.op import Constant, Parameter
from openvino.runtime import Extension
from openvino.runtime.exceptions import UserInputError
from openvino.runtime.utils.node_factory import NodeFactory
from openvino.runtime.utils.types import get_element_type

from openvino.runtime.passes import Manager, ConstantFolding
from tests.test_transformations.utils.utils import count_ops, PatternReplacement
from tests.test_graph.util import count_ops_of_type
from tests.test_transformations.utils.utils import get_relu_model as get_relu_transformations_model, MyModelPass

from tests.utils.helpers import (
    generate_image,
    generate_add_model,
    get_relu_model,
    create_filenames_for_ir,
    generate_abs_compiled_model_with_data,
)


def test_no_warning():
    with warnings.catch_warnings(record=True) as w:
        import openvino

        data = np.array([1, 2, 3])
        axis_vector = openvino.AxisVector(data)
        assert np.equal(axis_vector, data).all()

        assert len(w) == 0  # No warning


# request - https://docs.pytest.org/en/7.1.x/reference/reference.html#request
def test_read_model_from_ir(request, tmp_path):
    core = Core()
    xml_path, bin_path = create_filenames_for_ir(request.node.name, tmp_path)
    relu_model = get_relu_model()
    serialize(relu_model, xml_path, bin_path)
    model = core.read_model(model=xml_path, weights=bin_path)
    assert isinstance(model, Model)

    model = core.read_model(model=xml_path)
    assert isinstance(model, Model)


# request - https://docs.pytest.org/en/7.1.x/reference/reference.html#request
def test_read_model_as_path(request, tmp_path):
    core = Core()
    xml_path, bin_path = create_filenames_for_ir(request.node.name, tmp_path, True, True)
    relu_model = get_relu_model()
    serialize(relu_model, xml_path, bin_path)

    model = core.read_model(model=Path(xml_path), weights=Path(bin_path))
    assert isinstance(model, Model)

    model = core.read_model(model=xml_path, weights=Path(bin_path))
    assert isinstance(model, Model)

    model = core.read_model(model=Path(xml_path))
    assert isinstance(model, Model)


def test_infer_new_request_return_type(device):
    core = Core()
    model = get_relu_model()
    img = generate_image()
    compiled_model = core.compile_model(model, device)
    res = compiled_model.infer_new_request({"data": img})
    arr = res[list(res)[0]][0]

    assert isinstance(arr, np.ndarray)
    assert arr.itemsize == 4
    assert arr.shape == (3, 32, 32)
    assert arr.dtype == "float32"
    assert arr.nbytes == 12288


@pytest.mark.parametrize(("ov_type", "numpy_dtype"), [
    (Type.f32, np.float32),
    (Type.f64, np.float64),
    (Type.f16, np.float16),
    (Type.i8, np.int8),
    (Type.u8, np.uint8),
    (Type.i32, np.int32),
    (Type.i16, np.int16),
    (Type.u16, np.uint16),
    (Type.i64, np.int64),
])
@pytest.mark.parametrize("share_inputs", [True, False])
def test_infer_single_input(device, ov_type, numpy_dtype, share_inputs):
    _, request, tensor1, array1 = generate_abs_compiled_model_with_data(device, ov_type, numpy_dtype)

    request.infer(array1, share_inputs=share_inputs)
    assert np.array_equal(request.get_output_tensor().data, np.abs(array1))

    request.infer(tensor1, share_inputs=share_inputs)
    assert np.array_equal(request.get_output_tensor().data, np.abs(tensor1.data))


def test_infer_dynamic_model(device):
    core = Core()

    param = ops.parameter(PartialShape([-1, -1]))
    model = Model(ops.relu(param), [param])
    compiled_model = core.compile_model(model, device)
    assert compiled_model.input().partial_shape.is_dynamic
    request = compiled_model.create_infer_request()

    shape1 = [1, 28]
    request.infer([np.random.normal(size=shape1)])
    assert request.get_input_tensor().shape == Shape(shape1)

    shape2 = [1, 32]
    request.infer([np.random.normal(size=shape2)])
    assert request.get_input_tensor().shape == Shape(shape2)

    shape3 = [1, 40]
    request.infer(np.random.normal(size=shape3))
    assert request.get_input_tensor().shape == Shape(shape3)


def test_add_extension():
    class EmptyExtension(Extension):
        def __init__(self) -> None:
            super().__init__()

    core = Core()
    core.add_extension(EmptyExtension())
    core.add_extension([EmptyExtension(), EmptyExtension()])
    model = get_relu_model()
    assert isinstance(model, Model)


def test_output_replace():
    param = ops.parameter(PartialShape([1, 3, 22, 22]), name="parameter")
    relu = ops.relu(param.output(0))
    res = ops.result(relu.output(0), name="result")

    exp = ops.exp(param.output(0))
    relu.output(0).replace(exp.output(0))

    assert res.input_value(0).get_node() == exp


@pytest.mark.parametrize("share_inputs", [True, False])
def test_infer_queue(device, share_inputs):
    jobs = 8
    num_request = 4
    core = Core()
    model = get_relu_model()
    compiled_model = core.compile_model(model, device)
    infer_queue = AsyncInferQueue(compiled_model, num_request)
    jobs_done = [{"finished": False, "latency": 0} for _ in range(jobs)]

    def callback(request, job_id):
        jobs_done[job_id]["finished"] = True
        jobs_done[job_id]["latency"] = request.latency

    img = None

    if not share_inputs:
        img = generate_image()
    infer_queue.set_callback(callback)
    assert infer_queue.is_ready()

    for i in range(jobs):
        if share_inputs:
            img = generate_image()
        infer_queue.start_async({"data": img}, i, share_inputs=share_inputs)
    infer_queue.wait_all()
    assert all(job["finished"] for job in jobs_done)
    assert all(job["latency"] > 0 for job in jobs_done)


def test_model_reshape(device):
    shape = Shape([1, 10])
    param = ops.parameter(shape, dtype=np.float32)
    model = Model(ops.relu(param), [param])
    ref_shape = model.input().partial_shape
    ref_shape[0] = 3
    model.reshape(ref_shape)
    core = Core()
    compiled_model = core.compile_model(model, device)
    assert compiled_model.input().partial_shape == ref_shape


def test_model_get_raw_address():
    model = generate_add_model()
    model_with_same_addr = model
    model_different = generate_add_model()

    assert model._get_raw_address() == model_with_same_addr._get_raw_address()
    assert model._get_raw_address() != model_different._get_raw_address()


@pytest.mark.parametrize(
    ("ov_type", "numpy_dtype"),
    [
        (ov.Type.f32, np.float32),
        (ov.Type.f64, np.float64),
        (ov.Type.f16, np.float16),
        (ov.Type.bf16, np.float16),
        (ov.Type.i8, np.int8),
        (ov.Type.u8, np.uint8),
        (ov.Type.i32, np.int32),
        (ov.Type.u32, np.uint32),
        (ov.Type.i16, np.int16),
        (ov.Type.u16, np.uint16),
        (ov.Type.i64, np.int64),
        (ov.Type.u64, np.uint64),
        (ov.Type.boolean, bool),
    ],
)
def test_tensor_write_to_buffer(ov_type, numpy_dtype):
    ov_tensor = ov.Tensor(ov_type, ov.Shape([1, 3, 32, 32]))
    ones_arr = np.ones([1, 3, 32, 32], numpy_dtype)
    ov_tensor.data[:] = ones_arr
    assert np.array_equal(ov_tensor.data, ones_arr)


def test_strides_iteration_methods():
    data = np.array([1, 2, 3])
    strides = Strides(data)

    assert len(strides) == data.size
    assert np.equal(strides, data).all()
    assert np.equal([strides[i] for i in range(data.size)], data).all()

    data2 = np.array([5, 6, 7])
    for i in range(data2.size):
        strides[i] = data2[i]

    assert np.equal(strides, data2).all()


def test_node_factory_add():
    shape = [2, 2]
    dtype = np.int8
    parameter_a = ops8.parameter(shape, dtype=dtype, name="A")
    parameter_b = ops8.parameter(shape, dtype=dtype, name="B")

    factory = NodeFactory("opset1")
    arguments = NodeFactory._arguments_as_outputs([parameter_a, parameter_b])
    node = factory.create("Add", arguments, {})

    assert node.get_type_name() == "Add"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [2, 2]


def test_node_factory_validate_missing_arguments():
    factory = NodeFactory("opset1")

    try:
        factory.create(
            "TopK", None, {"axis": 1, "mode": "max", "sort": "value"},
        )
    except UserInputError:
        pass
    else:
        raise AssertionError("Validation of missing arguments has unexpectedly passed.")


@pytest.mark.parametrize(("const", "args", "expectation"), [
    (Constant, (Type.f32, Shape([3, 3]), list(range(9))), does_not_raise()),
    (ops8.constant, (np.arange(9).reshape(3, 3), Type.f32), does_not_raise()),
    (ops8.constant, (np.arange(9).reshape(3, 3), np.float32), does_not_raise()),
    (ops8.constant, [None], pytest.raises(ValueError)),
])
def test_constant(const, args, expectation):
    with expectation:
        node = const(*args)
        assert node.get_type_name() == "Constant"
        assert node.get_output_size() == 1
        assert list(node.get_output_shape(0)) == [3, 3]
        assert node.get_output_element_type(0) == Type.f32
        assert node.get_byte_size() == 36


def test_opset_reshape():
    element_type = Type.f32
    shape = Shape([2, 3])
    param1 = Parameter(element_type, shape)
    node = ops8.reshape(param1, Shape([3, 2]), special_zero=False)

    assert node.get_type_name() == "Reshape"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 2]
    assert node.get_output_element_type(0) == element_type


@pytest.mark.parametrize(
    ("input_shape", "dtype", "new_shape", "axis_mapping", "mode"),
    [
        ((3,), np.int32, [3, 3], [], []),
        ((4,), np.float32, [3, 4, 2, 4], [], []),
        ((3,), np.int8, [3, 3], [[0]], ["EXPLICIT"]),
    ],
)
def test_node_broadcast(input_shape, dtype, new_shape, axis_mapping, mode):
    input_data = ops.parameter(input_shape, name="input_data", dtype=dtype)
    node = ops.broadcast(input_data, new_shape, *axis_mapping, *mode)
    assert node.get_type_name() == "Broadcast"
    assert node.get_output_size() == 1
    assert node.get_output_element_type(0) == get_element_type(dtype)
    assert list(node.get_output_shape(0)) == new_shape


def test_model_pass():
    manager = Manager()
    model_pass = manager.register_pass(MyModelPass())
    manager.run_passes(get_relu_transformations_model())

    assert model_pass.model_changed


def test_runtime_graph_rewrite():
    import openvino.runtime.passes as rt
    model = get_relu_transformations_model()

    manager = rt.Manager()
    # check that register pass returns pass instance
    anchor = manager.register_pass(rt.GraphRewrite())
    anchor.add_matcher(PatternReplacement())
    manager.run_passes(model)

    assert count_ops(model, "Relu") == [2]


def test_runtime_passes_manager():
    node_constant = ops.constant(np.array([[0.0, 0.1, -0.1], [-2.5, 2.5, 3.0]], dtype=np.float32))
    node_ceil = ops.ceiling(node_constant)
    model = Model(node_ceil, [], "TestModel")

    assert count_ops_of_type(model, node_ceil) == 1
    assert count_ops_of_type(model, node_constant) == 1

    pass_manager = Manager()
    pass_manager.register_pass(ConstantFolding())
    pass_manager.run_passes(model)

    assert count_ops_of_type(model, node_ceil) == 0
    assert count_ops_of_type(model, node_constant) == 1


# from test_graph/test_ops_binary.py
@pytest.mark.parametrize(
    ("operator", "expected_type", "warning_type"),
    [
        (operator.add, Type.f32, warnings.catch_warnings(record=True)),
        (operator.sub, Type.f32, warnings.catch_warnings(record=True)),
        (operator.mul, Type.f32, warnings.catch_warnings(record=True)),
        (operator.truediv, Type.f32, warnings.catch_warnings(record=True)),
        (operator.eq, Type.boolean, pytest.warns(DeprecationWarning)),
        (operator.ne, Type.boolean, pytest.warns(DeprecationWarning)),
        (operator.gt, Type.boolean, pytest.warns(DeprecationWarning)),
        (operator.ge, Type.boolean, pytest.warns(DeprecationWarning)),
        (operator.lt, Type.boolean, pytest.warns(DeprecationWarning)),
        (operator.le, Type.boolean, pytest.warns(DeprecationWarning)),
    ],
)
def test_binary_operators(operator, expected_type, warning_type):
    value_b = np.array([[4, 5], [1, 7]], dtype=np.float32)

    shape = [2, 2]
    parameter_a = ops.parameter(shape, name="A", dtype=np.float32)

    with warning_type:
        model = operator(parameter_a, value_b)

    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == shape
    assert model.get_output_element_type(0) == expected_type


@pytest.mark.parametrize(
    ("operator", "expected_type", "warning_type"),
    [
        (operator.add, Type.f32, warnings.catch_warnings(record=True)),
        (operator.sub, Type.f32, warnings.catch_warnings(record=True)),
        (operator.mul, Type.f32, warnings.catch_warnings(record=True)),
        (operator.truediv, Type.f32, warnings.catch_warnings(record=True)),
        (operator.eq, Type.boolean, pytest.warns(DeprecationWarning)),
        (operator.ne, Type.boolean, pytest.warns(DeprecationWarning)),
        (operator.gt, Type.boolean, pytest.warns(DeprecationWarning)),
        (operator.ge, Type.boolean, pytest.warns(DeprecationWarning)),
        (operator.lt, Type.boolean, pytest.warns(DeprecationWarning)),
        (operator.le, Type.boolean, pytest.warns(DeprecationWarning)),
    ],
)
def test_binary_operators_with_scalar(operator, expected_type, warning_type):
    value_b = np.array([[5, 6], [7, 8]], dtype=np.float32)

    shape = [2, 2]
    parameter_a = ops.parameter(shape, name="A", dtype=np.float32)

    with warning_type:
        model = operator(parameter_a, value_b)

    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == shape
    assert model.get_output_element_type(0) == expected_type
