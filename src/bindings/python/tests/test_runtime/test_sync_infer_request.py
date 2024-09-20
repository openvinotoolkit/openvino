# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext as does_not_raise
from copy import deepcopy
import numpy as np
import os
import pytest
import datetime
import openvino.properties as props

import openvino.runtime.opset13 as ops
from openvino import (
    Core,
    CompiledModel,
    Model,
    Layout,
    PartialShape,
    Shape,
    Type,
    Tensor,
    compile_model,
)
from openvino.runtime import ProfilingInfo
from openvino.preprocess import PrePostProcessor

from tests.utils.helpers import (
    generate_image,
    get_relu_model,
    generate_concat_compiled_model_with_data,
    generate_add_compiled_model,
    generate_abs_compiled_model_with_data,
)


def create_simple_request_and_inputs(device):
    compiled_model = generate_add_compiled_model(device, input_shape=[2, 2])
    request = compiled_model.create_infer_request()

    arr_1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    arr_2 = np.array([[3, 4], [1, 2]], dtype=np.float32)

    return request, arr_1, arr_2


def test_get_profiling_info(device):
    core = Core()
    param = ops.parameter([1, 3, 32, 32], np.float32, name="data")
    softmax = ops.softmax(param, 1, name="fc_out")
    model = Model([softmax], [param], "test_model")

    core.set_property(device, {props.enable_profiling: True})
    compiled_model = core.compile_model(model, device)
    img = generate_image()
    request = compiled_model.create_infer_request()
    tensor_name = compiled_model.input("data").any_name
    request.infer({tensor_name: img})
    assert request.latency > 0
    prof_info = request.get_profiling_info()
    soft_max_node = next(node for node in prof_info if node.node_type == "Softmax")
    assert soft_max_node
    assert soft_max_node.status == ProfilingInfo.Status.EXECUTED
    assert isinstance(soft_max_node.real_time, datetime.timedelta)
    assert isinstance(soft_max_node.cpu_time, datetime.timedelta)
    assert isinstance(soft_max_node.exec_type, str)


def test_tensor_setter(device):
    core = Core()
    model = get_relu_model()

    compiled_1 = core.compile_model(model=model, device_name=device)
    compiled_2 = core.compile_model(model=model, device_name=device)
    compiled_3 = core.compile_model(model=model, device_name=device)

    img = generate_image()
    tensor = Tensor(img)

    request1 = compiled_1.create_infer_request()
    request1.set_tensor("data", tensor)
    t1 = request1.get_tensor("data")

    assert np.allclose(tensor.data, t1.data, atol=1e-2, rtol=1e-2)

    res = request1.infer({0: tensor})
    key = list(res)[0]
    res_1 = np.sort(res[key])
    t2 = request1.get_output_tensor()
    assert np.allclose(t2.data, res[key].data, atol=1e-2, rtol=1e-2)

    request = compiled_2.create_infer_request()
    res = request.infer({"data": tensor})
    res_2 = np.sort(request.get_output_tensor().data)
    assert np.allclose(res_1, res_2, atol=1e-2, rtol=1e-2)

    request.set_tensor("data", tensor)
    t3 = request.get_tensor("data")
    assert np.allclose(t3.data, t1.data, atol=1e-2, rtol=1e-2)

    request = compiled_3.create_infer_request()
    request.set_tensor(model.inputs[0], tensor)
    t1 = request1.get_tensor(model.inputs[0])

    assert np.allclose(tensor.data, t1.data, atol=1e-2, rtol=1e-2)

    res = request.infer()
    key = list(res)[0]
    res_1 = np.sort(res[key])
    t2 = request1.get_tensor(model.outputs[0])
    assert np.allclose(t2.data, res[key].data, atol=1e-2, rtol=1e-2)


def test_set_tensors(device):
    core = Core()

    param = ops.parameter([1, 3, 32, 32], np.float32, name="data")
    softmax = ops.softmax(param, 1, name="fc_out")
    res = ops.result(softmax, name="res")
    res.output(0).get_tensor().set_names({"res"})
    model = Model([res], [param], "test_model")

    compiled_model = core.compile_model(model, device)

    data1 = generate_image()
    tensor1 = Tensor(data1)
    data2 = np.ones(shape=(1, 3, 32, 32), dtype=np.float32)
    tensor2 = Tensor(data2)
    data3 = np.ones(shape=(1, 3, 32, 32), dtype=np.float32)
    tensor3 = Tensor(data3)
    data4 = np.zeros(shape=(1, 3, 32, 32), dtype=np.float32)
    tensor4 = Tensor(data4)

    request = compiled_model.create_infer_request()
    request.set_tensors({"data": tensor1, "res": tensor2})
    t1 = request.get_tensor("data")
    t2 = request.get_tensor("res")
    assert np.allclose(tensor1.data, t1.data, atol=1e-2, rtol=1e-2)
    assert np.allclose(tensor2.data, t2.data, atol=1e-2, rtol=1e-2)

    request.set_output_tensors({0: tensor2})
    output_node = compiled_model.outputs[0]
    t3 = request.get_tensor(output_node)
    assert np.allclose(tensor2.data, t3.data, atol=1e-2, rtol=1e-2)

    request.set_input_tensors({0: tensor1})
    output_node = compiled_model.inputs[0]
    t4 = request.get_tensor(output_node)
    assert np.allclose(tensor1.data, t4.data, atol=1e-2, rtol=1e-2)

    output_node = compiled_model.inputs[0]
    request.set_tensor(output_node, tensor3)
    t5 = request.get_tensor(output_node)
    assert np.allclose(tensor3.data, t5.data, atol=1e-2, rtol=1e-2)

    request.set_input_tensor(tensor3)
    t6 = request.get_tensor(request.model_inputs[0])
    assert np.allclose(tensor3.data, t6.data, atol=1e-2, rtol=1e-2)

    request.set_input_tensor(0, tensor1)
    t7 = request.get_tensor(request.model_inputs[0])
    assert np.allclose(tensor1.data, t7.data, atol=1e-2, rtol=1e-2)

    request.set_output_tensor(tensor2)
    t8 = request.get_tensor(request.model_outputs[0])
    assert np.allclose(tensor2.data, t8.data, atol=1e-2, rtol=1e-2)

    request.set_output_tensor(0, tensor4)
    t9 = request.get_tensor(request.model_outputs[0])
    assert np.allclose(tensor4.data, t9.data, atol=1e-2, rtol=1e-2)


def test_batched_tensors(device):
    core = Core()

    batch = 4
    one_shape = [1, 2, 2, 2]
    one_shape_size = np.prod(one_shape)
    batch_shape = [batch, 2, 2, 2]

    data1 = ops.parameter(batch_shape, np.float32)
    data1.set_friendly_name("input0")
    data1.get_output_tensor(0).set_names({"tensor_input0"})
    data1.set_layout(Layout("N..."))

    constant = ops.constant([1], np.float32)

    op1 = ops.add(data1, constant)
    op1.set_friendly_name("Add0")

    res1 = ops.result(op1)
    res1.set_friendly_name("Result0")
    res1.get_output_tensor(0).set_names({"tensor_output0"})

    model = Model([res1], [data1])

    compiled_model = core.compile_model(model, device)

    req = compiled_model.create_infer_request()

    # Allocate 8 chunks, set 'user tensors' to 0, 2, 4, 6 chunks
    buffer = np.zeros([batch * 2, *batch_shape[1:]], dtype=np.float32)

    tensors = []
    for i in range(batch):
        # non contiguous memory (i*2)
        tensors.append(Tensor(np.expand_dims(buffer[i * 2], 0), shared_memory=True))

    req.set_input_tensors(tensors)

    with pytest.raises(RuntimeError) as e:
        req.get_tensor("tensor_input0")
    assert "get_tensor shall not be used together with batched set_tensors/set_input_tensors" in str(e.value)

    actual_tensor = req.get_tensor("tensor_output0")
    actual = actual_tensor.data
    for test_num in range(0, 5):
        for i in range(0, batch):
            tensors[i].data[:] = test_num + 10

        req.infer()  # Adds '1' to each element

        # Reference values for each batch:
        _tmp = np.array([test_num + 11] * one_shape_size, dtype=np.float32).reshape([2, 2, 2])

        for idx in range(0, batch):
            assert np.array_equal(actual[idx], _tmp)


def test_inputs_outputs_property_and_method(device):
    num_inputs = 10
    input_shape = [1]
    params = [ops.parameter(input_shape, np.uint8) for _ in range(num_inputs)]
    model = Model(ops.split(ops.concat(params, 0), 0, num_inputs), params)
    core = Core()
    compiled_model = core.compile_model(model, device)
    request = compiled_model.create_infer_request()
    data = [np.atleast_1d(i) for i in range(num_inputs)]
    results = request.infer(data).values()
    for result, output_tensor in zip(results, request.output_tensors):
        assert np.array_equal(result, output_tensor.data)
    for input_data, input_tensor in zip(data, request.input_tensors):
        assert np.array_equal(input_data, input_tensor.data)
    for input_tensor in request.input_tensors:
        assert list(input_tensor.get_shape()) == input_shape
    for output_tensor in request.output_tensors:
        assert list(output_tensor.get_shape()) == input_shape


@pytest.mark.parametrize("share_inputs", [True, False])
def test_infer_list_as_inputs(device, share_inputs):
    num_inputs = 4
    input_shape = [2, 1]
    dtype = np.float32
    params = [ops.parameter(input_shape, dtype) for _ in range(num_inputs)]
    model = Model(ops.relu(ops.concat(params, 1)), params)
    core = Core()
    compiled_model = core.compile_model(model, device)

    def check_fill_inputs(request, inputs):
        for input_idx in range(len(inputs)):
            assert np.array_equal(request.get_input_tensor(input_idx).data, inputs[input_idx])

    request = compiled_model.create_infer_request()

    inputs = [np.random.normal(size=input_shape).astype(dtype)]
    request.infer(inputs, share_inputs=share_inputs)
    check_fill_inputs(request, inputs)

    inputs = [
        np.random.normal(size=input_shape).astype(dtype) for _ in range(num_inputs)
    ]
    request.infer(inputs, share_inputs=share_inputs)
    check_fill_inputs(request, inputs)


@pytest.mark.parametrize("share_inputs", [True, False])
def test_infer_mixed_keys(device, share_inputs):
    core = Core()
    model = get_relu_model()
    compiled_model = core.compile_model(model, device)

    img = generate_image()
    tensor = Tensor(img)

    data2 = np.ones(shape=img.shape, dtype=np.float32)
    tensor2 = Tensor(data2)

    request = compiled_model.create_infer_request()
    res = request.infer({0: tensor2, "data": tensor}, share_inputs=share_inputs)
    assert np.argmax(res[compiled_model.output()]) == 531


@pytest.mark.parametrize(("ov_type", "numpy_dtype"), [
    (Type.f32, np.float32),
    (Type.f64, np.float64),
    (Type.f16, np.float16),
    (Type.bf16, np.float16),
    (Type.i8, np.int8),
    (Type.u8, np.uint8),
    (Type.i32, np.int32),
    (Type.u32, np.uint32),
    (Type.i16, np.int16),
    (Type.u16, np.uint16),
    (Type.i64, np.int64),
    (Type.u64, np.uint64),
    (Type.boolean, bool),
])
@pytest.mark.parametrize("share_inputs", [True, False])
def test_infer_mixed_values(device, ov_type, numpy_dtype, share_inputs):
    request, tensor1, array1 = generate_concat_compiled_model_with_data(device=device, ov_type=ov_type, numpy_dtype=numpy_dtype)

    request.infer([tensor1, array1], share_inputs=share_inputs)

    assert np.array_equal(request.output_tensors[0].data, np.concatenate((tensor1.data, array1)))


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


def test_get_compiled_model(device):
    core = Core()
    param = ops.parameter([10])
    data = np.random.rand((10))
    model = Model(ops.relu(param), [param])
    compiled_model_1 = core.compile_model(model, device)
    infer_request = compiled_model_1.create_infer_request()
    compiled_model_2 = infer_request.get_compiled_model()

    ref = infer_request.infer({0: data})
    test = compiled_model_2.create_infer_request().infer({0: data})

    assert isinstance(compiled_model_2, CompiledModel)
    assert np.allclose(ref[0], test[0])


@pytest.mark.parametrize("share_inputs", [True, False])
def test_get_results(device, share_inputs):
    core = Core()
    data = ops.parameter([10], np.float64)
    model = Model(ops.split(data, 0, 5), [data])
    compiled_model = core.compile_model(model, device)
    request = compiled_model.create_infer_request()
    inputs = [np.random.normal(size=list(compiled_model.input().shape))]
    results = request.infer(inputs, share_inputs=share_inputs)
    for output in compiled_model.outputs:
        assert np.array_equal(results[output], request.results[output])


@pytest.mark.skipif(
    os.environ.get("TEST_DEVICE") not in ["GPU"],
    reason="Device dependent test",
)
@pytest.mark.parametrize("share_inputs", [True, False])
def test_infer_float16(device, share_inputs):
    model = bytes(
        b"""<net name="add_model" version="10">
    <layers>
    <layer id="0" name="x" type="Parameter" version="opset1">
        <data element_type="f16" shape="2,2,2"/>
        <output>
            <port id="0" precision="FP16">
                <dim>2</dim>
                <dim>2</dim>
                <dim>2</dim>
            </port>
        </output>
    </layer>
    <layer id="1" name="y" type="Parameter" version="opset1">
        <data element_type="f16" shape="2,2,2"/>
        <output>
            <port id="0" precision="FP16">
                <dim>2</dim>
                <dim>2</dim>
                <dim>2</dim>
            </port>
        </output>
    </layer>
    <layer id="2" name="sum" type="Add" version="opset1">
        <input>
            <port id="0">
                <dim>2</dim>
                <dim>2</dim>
                <dim>2</dim>
            </port>
            <port id="1">
                <dim>2</dim>
                <dim>2</dim>
                <dim>2</dim>
            </port>
        </input>
        <output>
            <port id="2" precision="FP16">
                <dim>2</dim>
                <dim>2</dim>
                <dim>2</dim>
            </port>
        </output>
    </layer>
    <layer id="3" name="sum/sink_port_0" type="Result" version="opset1">
        <input>
            <port id="0">
                <dim>2</dim>
                <dim>2</dim>
                <dim>2</dim>
            </port>
        </input>
    </layer>
    </layers>
    <edges>
    <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
    <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
    <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
    </edges>
</net>""")
    core = Core()
    model = core.read_model(model=model)
    ppp = PrePostProcessor(model)
    ppp.input(0).tensor().set_element_type(Type.f16)
    ppp.input(0).preprocess().convert_element_type(Type.f16)
    ppp.input(1).tensor().set_element_type(Type.f16)
    ppp.input(1).preprocess().convert_element_type(Type.f16)
    ppp.output(0).tensor().set_element_type(Type.f16)
    ppp.output(0).postprocess().convert_element_type(Type.f16)

    model = ppp.build()
    compiled_model = core.compile_model(model, device)
    input_data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(np.float16)
    request = compiled_model.create_infer_request()
    outputs = request.infer({0: input_data, 1: input_data}, share_inputs=share_inputs)
    assert np.allclose(list(outputs.values()), list(request.results.values()))
    assert np.allclose(list(outputs.values()), input_data + input_data)


@pytest.mark.parametrize("share_inputs", [True, False])
def test_ports_as_inputs(device, share_inputs):
    input_shape = [2, 2]
    param_a = ops.parameter(input_shape, np.float32)
    param_b = ops.parameter(input_shape, np.float32)
    model = Model(ops.add(param_a, param_b), [param_a, param_b])

    core = Core()
    compiled_model = core.compile_model(model, device)
    request = compiled_model.create_infer_request()

    arr_1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    arr_2 = np.array([[3, 4], [1, 2]], dtype=np.float32)

    tensor1 = Tensor(arr_1)
    tensor2 = Tensor(arr_2)

    res = request.infer(
        {compiled_model.inputs[0]: tensor1, compiled_model.inputs[1]: tensor2},
        share_inputs=share_inputs,
    )
    assert np.array_equal(res[compiled_model.outputs[0]], tensor1.data + tensor2.data)

    res = request.infer(
        {request.model_inputs[0]: tensor1, request.model_inputs[1]: tensor2},
        share_inputs=share_inputs,
    )
    assert np.array_equal(res[request.model_outputs[0]], tensor1.data + tensor2.data)


@pytest.mark.parametrize("share_inputs", [True, False])
def test_inputs_dict_not_replaced(device, share_inputs):
    request, arr_1, arr_2 = create_simple_request_and_inputs(device)

    inputs = {0: arr_1, 1: arr_2}
    inputs_copy = deepcopy(inputs)

    res = request.infer(inputs, share_inputs=share_inputs)

    np.testing.assert_equal(inputs, inputs_copy)
    assert np.array_equal(res[request.model_outputs[0]], arr_1 + arr_2)


@pytest.mark.parametrize("share_inputs", [True, False])
def test_inputs_list_not_replaced(device, share_inputs):
    request, arr_1, arr_2 = create_simple_request_and_inputs(device)

    inputs = [arr_1, arr_2]
    inputs_copy = deepcopy(inputs)

    res = request.infer(inputs, share_inputs=share_inputs)

    assert np.array_equal(inputs, inputs_copy)
    assert np.array_equal(res[request.model_outputs[0]], arr_1 + arr_2)


@pytest.mark.parametrize("share_inputs", [True, False])
def test_inputs_tuple_not_replaced(device, share_inputs):
    request, arr_1, arr_2 = create_simple_request_and_inputs(device)

    inputs = (arr_1, arr_2)
    inputs_copy = deepcopy(inputs)

    res = request.infer(inputs, share_inputs=share_inputs)

    assert np.array_equal(inputs, inputs_copy)
    assert np.array_equal(res[request.model_outputs[0]], arr_1 + arr_2)


@pytest.mark.parametrize("share_inputs", [True, False])
def test_invalid_inputs(device, share_inputs):
    request, _, _ = create_simple_request_and_inputs(device)

    class InvalidInput():
        pass

    inputs = InvalidInput()

    with pytest.raises(TypeError) as e:
        request.infer(inputs, share_inputs=share_inputs)
    assert "Incompatible inputs of type:" in str(e.value)


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


@pytest.mark.parametrize("share_inputs", [True, False])
def test_array_like_input_request(device, share_inputs):
    class ArrayLikeObject:
        # Array-like object accepted by np.array to test inputs similar to torch tensor and tf.Tensor
        def __init__(self, array) -> None:
            self.data = array

        def __array__(self):
            return np.array(self.data)

    _, request, _, input_data = generate_abs_compiled_model_with_data(device, Type.f32, np.single)
    model_input_object = ArrayLikeObject(input_data.tolist())
    model_input_list = [ArrayLikeObject(input_data.tolist())]
    model_input_dict = {0: ArrayLikeObject(input_data.tolist())}

    # Test single array-like object in InferRequest().Infer()
    res_object = request.infer(model_input_object, share_inputs=share_inputs)
    assert np.array_equal(res_object[request.model_outputs[0]], np.abs(input_data))

    # Test list of array-like objects to use normalize_inputs()
    res_list = request.infer(model_input_list)
    assert np.array_equal(res_list[request.model_outputs[0]], np.abs(input_data))

    # Test dict of array-like objects to use normalize_inputs()
    res_dict = request.infer(model_input_dict)
    assert np.array_equal(res_dict[request.model_outputs[0]], np.abs(input_data))


def test_convert_infer_request(device):
    request, arr_1, arr_2 = create_simple_request_and_inputs(device)
    inputs = [arr_1, arr_2]

    res = request.infer(inputs)
    with pytest.raises(TypeError) as e:
        deepcopy(res)
    assert "Cannot deepcopy 'openvino.runtime.ConstOutput' object." in str(e)


@pytest.mark.parametrize("share_inputs", [True, False])
@pytest.mark.parametrize("input_data", [
    np.array(1.0, dtype=np.float32),
    np.array(1, dtype=np.int32),
    np.float32(1.0),
    np.int32(1.0),
    1.0,
    1,
])
def test_only_scalar_infer(device, share_inputs, input_data):
    core = Core()
    param = ops.parameter([], np.float32, name="data")
    relu = ops.relu(param, name="relu")
    model = Model([relu], [param], "scalar_model")

    compiled = core.compile_model(model=model, device_name=device)
    request = compiled.create_infer_request()

    res = request.infer(input_data, share_inputs=share_inputs)

    assert res[request.model_outputs[0]] == np.maximum(input_data, 0)

    input_tensor = request.get_input_tensor()
    if share_inputs and isinstance(input_data, np.ndarray) and input_data.dtype == input_tensor.data.dtype:
        assert np.shares_memory(input_data, input_tensor.data)
    else:
        assert not np.shares_memory(input_data, input_tensor.data)


@pytest.mark.parametrize("share_inputs", [True, False])
@pytest.mark.parametrize("input_data", [
    {0: np.array(1.0, dtype=np.float32), 1: np.array([1.0, 2.0], dtype=np.float32)},
    {0: np.array(1, dtype=np.int32), 1: np.array([1, 2], dtype=np.int32)},
    {0: np.float32(1.0), 1: np.array([1, 2], dtype=np.float32)},
    {0: np.int32(1.0), 1: np.array([1, 2], dtype=np.int32)},
    {0: 1.0, 1: np.array([1.0, 2.0], dtype=np.float32)},
    {0: 1, 1: np.array([1.0, 2.0], dtype=np.int32)},
])
def test_mixed_scalar_infer(device, share_inputs, input_data):
    core = Core()
    param0 = ops.parameter([], np.float32, name="data0")
    param1 = ops.parameter([2], np.float32, name="data1")
    add = ops.add(param0, param1, name="add")
    model = Model([add], [param0, param1], "mixed_model")

    compiled = core.compile_model(model=model, device_name=device)
    request = compiled.create_infer_request()

    res = request.infer(input_data, share_inputs=share_inputs)

    assert np.allclose(res[request.model_outputs[0]], np.add(input_data[0], input_data[1]))

    input_tensor0 = request.get_input_tensor(0)
    input_tensor1 = request.get_input_tensor(1)

    if share_inputs:
        if isinstance(input_data[0], np.ndarray) and input_data[0].dtype == input_tensor0.data.dtype:
            assert np.shares_memory(input_data[0], input_tensor0.data)
        else:
            assert not np.shares_memory(input_data[0], input_tensor0.data)
        if isinstance(input_data[1], np.ndarray) and input_data[1].dtype == input_tensor1.data.dtype:
            assert np.shares_memory(input_data[1], input_tensor1.data)
        else:
            assert not np.shares_memory(input_data[1], input_tensor1.data)
    else:
        assert not np.shares_memory(input_data[0], input_tensor0.data)
        assert not np.shares_memory(input_data[1], input_tensor1.data)


@pytest.mark.parametrize("share_inputs", [True, False])
@pytest.mark.parametrize("input_data", [
    {0: np.array(1.0, dtype=np.float32), 1: np.array([3.0], dtype=np.float32)},
    {0: np.array(1.0, dtype=np.float32), 1: np.array([3.0, 3.0, 3.0], dtype=np.float32)},
])
def test_mixed_dynamic_infer(device, share_inputs, input_data):
    core = Core()
    param0 = ops.parameter([], np.float32, name="data0")
    param1 = ops.parameter(["?"], np.float32, name="data1")
    add = ops.add(param0, param1, name="add")
    model = Model([add], [param0, param1], "mixed_model")

    compiled = core.compile_model(model=model, device_name=device)
    request = compiled.create_infer_request()

    res = request.infer(input_data, share_inputs=share_inputs)

    assert np.allclose(res[request.model_outputs[0]], np.add(input_data[0], input_data[1]))

    input_tensor0 = request.get_input_tensor(0)
    input_tensor1 = request.get_input_tensor(1)

    if share_inputs:
        if isinstance(input_data[0], np.ndarray) and input_data[0].dtype == input_tensor0.data.dtype:
            assert np.shares_memory(input_data[0], input_tensor0.data)
        else:
            assert not np.shares_memory(input_data[0], input_tensor0.data)
        if isinstance(input_data[1], np.ndarray) and input_data[1].dtype == input_tensor1.data.dtype:
            assert np.shares_memory(input_data[1], input_tensor1.data)
        else:
            assert not np.shares_memory(input_data[1], input_tensor1.data)
    else:
        assert not np.shares_memory(input_data[0], input_tensor0.data)
        assert not np.shares_memory(input_data[1], input_tensor1.data)


@pytest.mark.parametrize("share_inputs", [True, False])
@pytest.mark.parametrize(("input_data", "change_flags"), [
    ({0: np.frombuffer(b"\x01\x02\x03\x04", np.uint8)}, False),
    ({0: np.array([1, 2, 3, 4], dtype=np.uint8)}, True),
])
def test_not_writable_inputs_infer(device, share_inputs, input_data, change_flags):
    if change_flags is True:
        input_data[0].setflags(write=0)
    # identity model
    input_shape = [4]
    param_node = ops.parameter(input_shape, np.uint8, name="data0")
    core = Core()
    model = Model(param_node, [param_node])
    compiled = core.compile_model(model, "CPU")

    results = compiled(input_data, share_inputs=share_inputs)

    assert np.array_equal(results[0], input_data[0])

    request = compiled.create_infer_request()
    results = request.infer(input_data, share_inputs=share_inputs)

    assert np.array_equal(results[0], input_data[0])

    input_tensor = request.get_input_tensor(0)

    # Not writable inputs should always be copied.
    assert not np.shares_memory(input_data[0], input_tensor.data)


@pytest.mark.parametrize("share_inputs", [True, False])
@pytest.mark.parametrize("share_outputs", [True, False])
@pytest.mark.parametrize("is_positional", [True, False])
def test_compiled_model_share_memory(device, share_inputs, share_outputs, is_positional):
    compiled, _, _, input_data = generate_abs_compiled_model_with_data(device, Type.f32, np.float32)

    if is_positional:
        results = compiled(input_data, share_inputs=share_inputs, share_outputs=share_outputs)
    else:
        results = compiled(input_data, share_inputs, share_outputs)

    assert np.array_equal(results[0], np.abs(input_data))

    in_tensor_shares = np.shares_memory(compiled._infer_request.get_input_tensor(0).data, input_data)
    if share_inputs:
        assert in_tensor_shares
    else:
        assert not in_tensor_shares

    out_tensor_shares = np.shares_memory(compiled._infer_request.get_output_tensor(0).data, results[0])
    if share_outputs:
        assert out_tensor_shares
        assert results[0].flags["OWNDATA"] is False
    else:
        assert not out_tensor_shares
        assert results[0].flags["OWNDATA"] is True


@pytest.mark.parametrize("share_inputs", [True, False])
@pytest.mark.parametrize("share_outputs", [True, False])
@pytest.mark.parametrize("is_positional", [True, False])
def test_infer_request_share_memory(device, share_inputs, share_outputs, is_positional):
    _, request, _, input_data = generate_abs_compiled_model_with_data(device, Type.f32, np.float32)

    if is_positional:
        results = request.infer(input_data, share_inputs=share_inputs, share_outputs=share_outputs)
    else:
        results = request.infer(input_data, share_inputs, share_outputs)

    assert np.array_equal(results[0], np.abs(input_data))

    in_tensor_shares = np.shares_memory(request.get_input_tensor(0).data, input_data)

    if share_inputs:
        assert in_tensor_shares
    else:
        assert not in_tensor_shares

    out_tensor_shares = np.shares_memory(request.get_output_tensor(0).data, results[0])
    if share_outputs:
        assert out_tensor_shares
        assert results[0].flags["OWNDATA"] is False
    else:
        assert not out_tensor_shares
        assert results[0].flags["OWNDATA"] is True


def test_output_result_to_input():
    def create_model_1():
        param1 = ops.parameter(Shape([1, 1]), Type.i32)
        param1.set_friendly_name("input_1")
        add = ops.add(param1, ops.constant([1], Type.i32))
        add1 = ops.add(param1, ops.constant([[5]], Type.i32))
        model = Model([add, add1], [param1])
        model.output(0).tensor.set_names({"output_1_1"})
        model.output(1).tensor.set_names({"outputs_1_2"})
        return model

    def create_model_2():
        param1 = ops.parameter(Shape([1, 1]), Type.i32)
        param1.set_friendly_name("output_1_1")
        param2 = ops.parameter(Shape([1, 1]), Type.i32)
        param2.set_friendly_name("outputs_1_2")

        add = ops.add(param1, param2)
        model = Model([add], [param1, param2])
        model.output(0).tensor.set_names({"output_2_1"})
        return model

    model_1 = create_model_1()
    model_2 = create_model_2()
    compiled_1, compiled_2 = compile_model(model_1), compile_model(model_2)
    input_data = np.array([[1]])
    result_1 = compiled_1(input_data, share_inputs=False)
    with does_not_raise():
        result_2 = compiled_2(result_1, share_inputs=False)
    assert np.array_equal(result_2[0], [[8]])
