# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
import numpy as np
import os
import pytest
import datetime
import time

import openvino.runtime.opset8 as ops
from openvino.runtime import Core, AsyncInferQueue, Tensor, ProfilingInfo, Model
from openvino.runtime import Type, PartialShape, Shape, Layout
from openvino.preprocess import PrePostProcessor

# TODO: reformat into absolute paths
from ..conftest import model_path
from ..test_utils.test_utils import generate_image

is_myriad = os.environ.get("TEST_DEVICE") == "MYRIAD"
test_net_xml, test_net_bin = model_path(is_myriad)


def create_model_with_memory(input_shape, data_type):
    input_data = ops.parameter(input_shape, name="input_data", dtype=data_type)
    rv = ops.read_value(input_data, "var_id_667")
    add = ops.add(rv, input_data, name="MemoryAdd")
    node = ops.assign(add, "var_id_667")
    res = ops.result(add, "res")
    model = Model(results=[res], sinks=[node], parameters=[input_data], name="name")
    return model


def create_simple_request_and_inputs(device):
    input_shape = [2, 2]
    param_a = ops.parameter(input_shape, np.float32)
    param_b = ops.parameter(input_shape, np.float32)
    model = Model(ops.add(param_a, param_b), [param_a, param_b])

    core = Core()
    compiled_model = core.compile_model(model, device)
    request = compiled_model.create_infer_request()

    arr_1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    arr_2 = np.array([[3, 4], [1, 2]], dtype=np.float32)

    return request, arr_1, arr_2


def concat_model_with_data(device, ov_type, numpy_dtype):
    input_shape = [5]

    params = []
    params += [ops.parameter(input_shape, ov_type)]
    if ov_type == Type.bf16:
        params += [ops.parameter(input_shape, ov_type)]
    else:
        params += [ops.parameter(input_shape, numpy_dtype)]

    model = Model(ops.concat(params, 0), params)
    core = Core()
    compiled_model = core.compile_model(model, device)
    request = compiled_model.create_infer_request()
    tensor1 = Tensor(ov_type, input_shape)
    tensor1.data[:] = np.array([6, 7, 8, 9, 0])
    array1 = np.array([1, 2, 3, 4, 5], dtype=numpy_dtype)

    return request, tensor1, array1


def abs_model_with_data(device, ov_type, numpy_dtype):
    input_shape = [1, 4]
    param = ops.parameter(input_shape, ov_type)
    model = Model(ops.abs(param), [param])
    core = Core()
    compiled_model = core.compile_model(model, device)

    request = compiled_model.create_infer_request()

    tensor1 = Tensor(ov_type, input_shape)
    tensor1.data[:] = np.array([6, -7, -8, 9])

    array1 = np.array([[-1, 2, 5, -3]]).astype(numpy_dtype)

    return request, tensor1, array1


def test_get_profiling_info(device):
    core = Core()
    model = core.read_model(test_net_xml, test_net_bin)
    core.set_property(device, {"PERF_COUNT": "YES"})
    compiled_model = core.compile_model(model, device)
    img = generate_image()
    request = compiled_model.create_infer_request()
    tensor_name = compiled_model.input("data").any_name
    request.infer({tensor_name: img})
    assert request.latency > 0
    prof_info = request.get_profiling_info()
    soft_max_node = next(node for node in prof_info if node.node_name == "fc_out")
    assert soft_max_node.node_type == "Softmax"
    assert soft_max_node.status == ProfilingInfo.Status.EXECUTED
    assert isinstance(soft_max_node.real_time, datetime.timedelta)
    assert isinstance(soft_max_node.cpu_time, datetime.timedelta)
    assert isinstance(soft_max_node.exec_type, str)


def test_tensor_setter(device):
    core = Core()
    model = core.read_model(test_net_xml, test_net_bin)
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
    t2 = request1.get_tensor("fc_out")
    assert np.allclose(t2.data, res[key].data, atol=1e-2, rtol=1e-2)

    request = compiled_2.create_infer_request()
    res = request.infer({"data": tensor})
    res_2 = np.sort(request.get_tensor("fc_out").data)
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
    model = core.read_model(test_net_xml, test_net_bin)
    compiled_model = core.compile_model(model, device)

    data1 = generate_image()
    tensor1 = Tensor(data1)
    data2 = np.ones(shape=(1, 10), dtype=np.float32)
    tensor2 = Tensor(data2)
    data3 = np.ones(shape=(1, 3, 32, 32), dtype=np.float32)
    tensor3 = Tensor(data3)
    data4 = np.zeros(shape=(1, 10), dtype=np.float32)
    tensor4 = Tensor(data4)

    request = compiled_model.create_infer_request()
    request.set_tensors({"data": tensor1, "fc_out": tensor2})
    t1 = request.get_tensor("data")
    t2 = request.get_tensor("fc_out")
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


def test_inputs_outputs_property(device):
    num_inputs = 10
    input_shape = [1]
    params = [ops.parameter(input_shape, np.uint8) for _ in range(num_inputs)]
    model = Model(ops.split(ops.concat(params, 0), 0, num_inputs), params)
    core = Core()
    compiled_model = core.compile_model(model, device)
    request = compiled_model.create_infer_request()
    data = [np.atleast_1d(i) for i in range(num_inputs)]
    results = request.infer(data).values()
    for result, output_tensor in zip(results, request.outputs):
        assert np.array_equal(result, output_tensor.data)
    for input_data, input_tensor in zip(data, request.inputs):
        assert np.array_equal(input_data, input_tensor.data)


def test_cancel(device):
    core = Core()
    model = core.read_model(test_net_xml, test_net_bin)
    compiled_model = core.compile_model(model, device)
    img = generate_image()
    request = compiled_model.create_infer_request()

    request.start_async({0: img})
    request.cancel()
    with pytest.raises(RuntimeError) as e:
        request.wait()
    assert "[ INFER_CANCELLED ]" in str(e.value)

    request.start_async({"data": img})
    request.cancel()
    with pytest.raises(RuntimeError) as e:
        request.wait_for(1)
    assert "[ INFER_CANCELLED ]" in str(e.value)


def test_start_async(device):
    core = Core()
    model = core.read_model(test_net_xml, test_net_bin)
    compiled_model = core.compile_model(model, device)
    img = generate_image()
    jobs = 3
    requests = []
    for _ in range(jobs):
        requests.append(compiled_model.create_infer_request())

    def callback(callbacks_info):
        time.sleep(0.01)
        callbacks_info["finished"] += 1

    callbacks_info = {}
    callbacks_info["finished"] = 0
    for request in requests:
        request.set_callback(callback, callbacks_info)
        request.start_async({0: img})
    for request in requests:
        request.wait()
        assert request.latency > 0
    assert callbacks_info["finished"] == jobs


def test_infer_list_as_inputs(device):
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
    request.infer(inputs)
    check_fill_inputs(request, inputs)

    inputs = [
        np.random.normal(size=input_shape).astype(dtype) for _ in range(num_inputs)
    ]
    request.infer(inputs)
    check_fill_inputs(request, inputs)


def test_infer_mixed_keys(device):
    core = Core()
    model = core.read_model(test_net_xml, test_net_bin)
    core.set_property(device, {"PERF_COUNT": "YES"})
    model = core.compile_model(model, device)

    img = generate_image()
    tensor = Tensor(img)

    data2 = np.ones(shape=img.shape, dtype=np.float32)
    tensor2 = Tensor(data2)

    request = model.create_infer_request()
    res = request.infer({0: tensor2, "data": tensor})
    assert np.argmax(res[model.output()]) == 9


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
def test_infer_mixed_values(device, ov_type, numpy_dtype):
    request, tensor1, array1 = concat_model_with_data(device, ov_type, numpy_dtype)

    request.infer([tensor1, array1])

    assert np.array_equal(request.outputs[0].data, np.concatenate((tensor1.data, array1)))


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
def test_async_mixed_values(device, ov_type, numpy_dtype):
    request, tensor1, array1 = concat_model_with_data(device, ov_type, numpy_dtype)

    request.start_async([tensor1, array1])
    request.wait()

    assert np.array_equal(request.outputs[0].data, np.concatenate((tensor1.data, array1)))


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
def test_infer_single_input(device, ov_type, numpy_dtype):
    request, tensor1, array1 = abs_model_with_data(device, ov_type, numpy_dtype)

    request.infer(array1)
    assert np.array_equal(request.get_output_tensor().data, np.abs(array1))

    request.infer(tensor1)
    assert np.array_equal(request.get_output_tensor().data, np.abs(tensor1.data))


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
def test_async_single_input(device, ov_type, numpy_dtype):
    request, tensor1, array1 = abs_model_with_data(device, ov_type, numpy_dtype)

    request.start_async(array1)
    request.wait()
    assert np.array_equal(request.get_output_tensor().data, np.abs(array1))

    request.start_async(tensor1)
    request.wait()
    assert np.array_equal(request.get_output_tensor().data, np.abs(tensor1.data))


def test_infer_queue(device):
    jobs = 8
    num_request = 4
    core = Core()
    model = core.read_model(test_net_xml, test_net_bin)
    compiled_model = core.compile_model(model, device)
    infer_queue = AsyncInferQueue(compiled_model, num_request)
    jobs_done = [{"finished": False, "latency": 0} for _ in range(jobs)]

    def callback(request, job_id):
        jobs_done[job_id]["finished"] = True
        jobs_done[job_id]["latency"] = request.latency

    img = generate_image()
    infer_queue.set_callback(callback)
    for i in range(jobs):
        infer_queue.start_async({"data": img}, i)
    infer_queue.wait_all()
    assert all(job["finished"] for job in jobs_done)
    assert all(job["latency"] > 0 for job in jobs_done)


def test_infer_queue_is_ready(device):
    core = Core()
    param = ops.parameter([10])
    model = Model(ops.relu(param), [param])
    compiled_model = core.compile_model(model, device)
    infer_queue = AsyncInferQueue(compiled_model, 1)

    def callback(request, _):
        time.sleep(0.001)

    infer_queue.set_callback(callback)
    assert infer_queue.is_ready()
    infer_queue.start_async()
    assert not infer_queue.is_ready()
    infer_queue.wait_all()


def test_infer_queue_fail_on_cpp_model(device):
    jobs = 6
    num_request = 4
    core = Core()
    model = core.read_model(test_net_xml, test_net_bin)
    compiled_model = core.compile_model(model, device)
    infer_queue = AsyncInferQueue(compiled_model, num_request)

    def callback(request, _):
        request.get_tensor("Unknown")

    img = generate_image()
    infer_queue.set_callback(callback)

    with pytest.raises(RuntimeError) as e:
        for _ in range(jobs):
            infer_queue.start_async({"data": img})
        infer_queue.wait_all()

    assert "Port for tensor name Unknown was not found" in str(e.value)


def test_infer_queue_fail_on_py_model(device):
    jobs = 1
    num_request = 1
    core = Core()
    model = core.read_model(test_net_xml, test_net_bin)
    compiled_model = core.compile_model(model, device)
    infer_queue = AsyncInferQueue(compiled_model, num_request)

    def callback(request, _):
        request = request + 21

    img = generate_image()
    infer_queue.set_callback(callback)

    with pytest.raises(TypeError) as e:
        for _ in range(jobs):
            infer_queue.start_async({"data": img})
        infer_queue.wait_all()

    assert "unsupported operand type(s) for +" in str(e.value)


@pytest.mark.parametrize("with_callback", [False, True])
def test_infer_queue_fail_in_inference(device, with_callback):
    jobs = 6
    num_request = 4
    core = Core()
    data = ops.parameter([5, 2], dtype=np.float32, name="data")
    indexes = ops.parameter(Shape([3, 2]), dtype=np.int32, name="indexes")
    emb = ops.embedding_bag_packed_sum(data, indexes)
    model = Model(emb, [data, indexes])
    compiled_model = core.compile_model(model, device)
    infer_queue = AsyncInferQueue(compiled_model, num_request)

    def callback(request, _):
        pytest.fail("Callback should not be called")

    if with_callback:
        infer_queue.set_callback(callback)

    data_tensor = Tensor(np.arange(10).reshape((5, 2)).astype(np.float32))
    indexes_tensor = Tensor(np.array([[100, 101], [102, 103], [104, 105]], dtype=np.int32))

    with pytest.raises(RuntimeError) as e:
        for _ in range(jobs):
            infer_queue.start_async({"data": data_tensor, "indexes": indexes_tensor})
        infer_queue.wait_all()

    assert "has invalid embedding bag index:" in str(e.value)


def test_infer_queue_get_idle_handle(device):
    param = ops.parameter([10])
    model = Model(ops.relu(param), [param])
    core = Core()
    compiled_model = core.compile_model(model, device)
    queue = AsyncInferQueue(compiled_model, 2)
    niter = 10

    for _ in range(len(queue)):
        queue.start_async()
    queue.wait_all()
    for request in queue:
        assert request.wait_for(0)

    for _ in range(niter):
        idle_id = queue.get_idle_request_id()
        assert queue[idle_id].wait_for(0)
        queue.start_async()
    queue.wait_all()


@pytest.mark.parametrize("data_type",
                         [np.float32,
                          np.int32,
                          np.float16])
@pytest.mark.parametrize("mode", ["set_init_memory_state", "reset_memory_state", "normal"])
@pytest.mark.parametrize("input_shape", [[10], [10, 10], [10, 10, 10], [2, 10, 10, 10]])
@pytest.mark.skipif(
    os.environ.get("TEST_DEVICE", "CPU") != "CPU",
    reason=f"Can't run test on device {os.environ.get('TEST_DEVICE', 'CPU')}, "
    "Memory layers fully supported only on CPU",
)
def test_query_state_write_buffer(device, input_shape, data_type, mode):
    core = Core()
    if device == "CPU":
        if core.get_property(device, "FULL_DEVICE_NAME") == "arm_compute::NEON":
            pytest.skip("Can't run on ARM plugin")

    from openvino.runtime import Tensor
    from openvino.runtime.utils.types import get_dtype

    model = create_model_with_memory(input_shape, data_type)
    compiled_model = core.compile_model(model=model, device_name=device)
    request = compiled_model.create_infer_request()
    mem_states = request.query_state()
    mem_state = mem_states[0]

    assert mem_state.name == "var_id_667"
    # todo: Uncomment after fix 45611,
    #  CPU plugin returns outputs and memory state in FP32 in case of FP16 original precision
    # Code: assert mem_state.state.tensor_desc.precision == data_type

    for i in range(1, 10):
        if mode == "set_init_memory_state":
            # create initial value
            const_init = 5
            init_array = np.full(input_shape, const_init, dtype=get_dtype(mem_state.state.element_type))
            tensor = Tensor(init_array)
            mem_state.state = tensor

            res = request.infer({0: np.full(input_shape, 1, dtype=data_type)})
            expected_res = np.full(input_shape, 1 + const_init, dtype=data_type)
        elif mode == "reset_memory_state":
            # reset initial state of ReadValue to zero
            mem_state.reset()
            res = request.infer({0: np.full(input_shape, 1, dtype=data_type)})
            # always ones
            expected_res = np.full(input_shape, 1, dtype=data_type)
        else:
            res = request.infer({0: np.full(input_shape, 1, dtype=data_type)})
            expected_res = np.full(input_shape, i, dtype=data_type)

        assert np.allclose(res[list(res)[0]], expected_res, atol=1e-6), f"Expected values: {expected_res} \n Actual values: {res} \n"


def test_get_results(device):
    core = Core()
    data = ops.parameter([10], np.float64)
    model = Model(ops.split(data, 0, 5), [data])
    compiled_model = core.compile_model(model, device)
    request = compiled_model.create_infer_request()
    inputs = [np.random.normal(size=list(compiled_model.input().shape))]
    results = request.infer(inputs)
    for output in compiled_model.outputs:
        assert np.array_equal(results[output], request.results[output])


def test_results_async_infer(device):
    jobs = 8
    num_request = 4
    core = Core()
    model = core.read_model(test_net_xml, test_net_bin)
    compiled_model = core.compile_model(model, device)
    infer_queue = AsyncInferQueue(compiled_model, num_request)
    jobs_done = [{"finished": False, "latency": 0} for _ in range(jobs)]

    def callback(request, job_id):
        jobs_done[job_id]["finished"] = True
        jobs_done[job_id]["latency"] = request.latency

    img = generate_image()
    infer_queue.set_callback(callback)
    for i in range(jobs):
        infer_queue.start_async({"data": img}, i)
    infer_queue.wait_all()

    request = compiled_model.create_infer_request()
    outputs = request.infer({0: img})

    for i in range(num_request):
        assert np.allclose(list(outputs.values()), list(infer_queue[i].results.values()))


@pytest.mark.skipif(
    os.environ.get("TEST_DEVICE") not in ["GPU, FPGA", "MYRIAD"],
    reason="Device independent test",
)
def test_infer_float16(device):
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
    outputs = request.infer({0: input_data, 1: input_data})
    assert np.allclose(list(outputs.values()), list(request.results.values()))
    assert np.allclose(list(outputs.values()), input_data + input_data)


def test_ports_as_inputs(device):
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

    res = request.infer({compiled_model.inputs[0]: tensor1, compiled_model.inputs[1]: tensor2})
    assert np.array_equal(res[compiled_model.outputs[0]], tensor1.data + tensor2.data)

    res = request.infer({request.model_inputs[0]: tensor1, request.model_inputs[1]: tensor2})
    assert np.array_equal(res[request.model_outputs[0]], tensor1.data + tensor2.data)


def test_inputs_dict_not_replaced(device):
    request, arr_1, arr_2 = create_simple_request_and_inputs(device)

    inputs = {0: arr_1, 1: arr_2}
    inputs_copy = deepcopy(inputs)

    res = request.infer(inputs)

    np.testing.assert_equal(inputs, inputs_copy)
    assert np.array_equal(res[request.model_outputs[0]], arr_1 + arr_2)


def test_inputs_list_not_replaced(device):
    request, arr_1, arr_2 = create_simple_request_and_inputs(device)

    inputs = [arr_1, arr_2]
    inputs_copy = deepcopy(inputs)

    res = request.infer(inputs)

    assert np.array_equal(inputs, inputs_copy)
    assert np.array_equal(res[request.model_outputs[0]], arr_1 + arr_2)


def test_inputs_tuple_not_replaced(device):
    request, arr_1, arr_2 = create_simple_request_and_inputs(device)

    inputs = (arr_1, arr_2)
    inputs_copy = deepcopy(inputs)

    res = request.infer(inputs)

    assert np.array_equal(inputs, inputs_copy)
    assert np.array_equal(res[request.model_outputs[0]], arr_1 + arr_2)


def test_invalid_inputs(device):
    request, _, _ = create_simple_request_and_inputs(device)

    inputs = "some_input"

    with pytest.raises(TypeError) as e:
        request.infer(inputs)
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


def test_array_like_input_request(device):
    class ArrayLikeObject:
        # Array-like object accepted by np.array to test inputs similar to torch tensor and tf.Tensor
        def __init__(self, array) -> None:
            self.data = array

        def __array__(self):
            return np.array(self.data)

    request, _, input_data = abs_model_with_data(device, Type.f32, np.single)
    model_input_object = ArrayLikeObject(input_data.tolist())
    model_input_list = [ArrayLikeObject(input_data.tolist())]

    # Test single array-like object in InferRequest().Infer()
    res_object = request.infer(model_input_object)
    assert np.array_equal(res_object[request.model_outputs[0]], np.abs(input_data))

    # Test list of array-like objects to use normalize_inputs()
    res_list = request.infer(model_input_list)
    assert np.array_equal(res_list[request.model_outputs[0]], np.abs(input_data))


def test_array_like_input_async(device):
    class ArrayLikeObject:
        # Array-like object accepted by np.array to test inputs similar to torch tensor and tf.Tensor
        def __init__(self, array) -> None:
            self.data = array

        def __array__(self):
            return np.array(self.data)

    request, _, input_data = abs_model_with_data(device, Type.f32, np.single)
    model_input_object = ArrayLikeObject(input_data.tolist())
    model_input_list = [ArrayLikeObject(input_data.tolist())]
    # Test single array-like object in InferRequest().start_async()
    request.start_async(model_input_object)
    request.wait()
    assert np.array_equal(request.get_output_tensor().data, np.abs(input_data))

    # Test list of array-like objects in InferRequest().start_async()
    request.start_async(model_input_list)
    request.wait()
    assert np.array_equal(request.get_output_tensor().data, np.abs(input_data))


def test_array_like_input_async_infer_queue(device):
    class ArrayLikeObject:
        # Array-like object accepted by np.array to test inputs similar to torch tensor and tf.Tensor
        def __init__(self, array) -> None:
            self.data = array

        def __array__(self):
            return np.array(self.data)

    jobs = 8
    ov_type = Type.f32
    input_shape = [2, 2]
    input_data = [[-2, -1], [0, 1]]
    param = ops.parameter(input_shape, ov_type)
    layer = ops.abs(param)
    model = Model([layer], [param])
    core = Core()
    compiled_model = core.compile_model(model, "CPU")

    model_input_object = ArrayLikeObject(input_data)
    model_input_list = [ArrayLikeObject(input_data)]

    # Test single array-like object in AsyncInferQueue.start_async()
    infer_queue_object = AsyncInferQueue(compiled_model, jobs)
    for _i in range(jobs):
        infer_queue_object.start_async(model_input_object)
    infer_queue_object.wait_all()
    for i in range(jobs):
        assert np.array_equal(infer_queue_object[i].get_output_tensor().data, np.abs(input_data))

    # Test list of array-like objects in AsyncInferQueue.start_async()
    infer_queue_list = AsyncInferQueue(compiled_model, jobs)
    for _i in range(jobs):
        infer_queue_list.start_async(model_input_list)
    infer_queue_list.wait_all()
    for i in range(jobs):
        assert np.array_equal(infer_queue_list[i].get_output_tensor().data, np.abs(input_data))


def test_convert_infer_request(device):
    request, arr_1, arr_2 = create_simple_request_and_inputs(device)
    inputs = [arr_1, arr_2]

    res = request.infer(inputs)
    with pytest.raises(TypeError) as e:
        deepcopy(res)
    assert "cannot deepcopy 'openvino.runtime.ConstOutput' object." in str(e)
