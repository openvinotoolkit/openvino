# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from copy import deepcopy
import numpy as np
import pytest

import openvino.runtime.opset13 as ops
from openvino import (
    Core,
    InferRequest,
    AsyncInferQueue,
    Model,
    Shape,
    Type,
    Tensor,
)
from tests import skip_need_mock_op
from tests.utils.helpers import generate_image, get_relu_model, generate_model_with_memory


def concat_model_with_data(device, ov_type, numpy_dtype):
    core = Core()

    input_shape = [5]

    params = []
    params += [ops.parameter(input_shape, ov_type)]
    if ov_type == Type.bf16:
        params += [ops.parameter(input_shape, ov_type)]
    else:
        params += [ops.parameter(input_shape, numpy_dtype)]

    model = Model(ops.concat(params, 0), params)
    compiled_model = core.compile_model(model, device)
    request = compiled_model.create_infer_request()
    tensor1 = Tensor(ov_type, input_shape)
    tensor1.data[:] = np.array([6, 7, 8, 9, 0])
    array1 = np.array([1, 2, 3, 4, 5], dtype=numpy_dtype)

    return request, tensor1, array1


def create_model_with_memory(input_shape, data_type):
    input_data = ops.parameter(input_shape, name="input_data", dtype=data_type)
    rv = ops.read_value(input_data, "var_id_667", data_type, input_shape)
    add = ops.add(rv, input_data, name="MemoryAdd")
    node = ops.assign(add, "var_id_667")
    res = ops.result(add, "res")
    model = Model(results=[res], sinks=[node], parameters=[input_data], name="name")
    return model


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

    return compiled_model, request, tensor1, array1


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


def test_infer_queue_iteration(device):
    core = Core()
    param = ops.parameter([10])
    model = Model(ops.relu(param), [param])
    compiled_model = core.compile_model(model, device)
    infer_queue = AsyncInferQueue(compiled_model, 1)
    assert isinstance(infer_queue, Iterable)
    for infer_req in infer_queue:
        assert isinstance(infer_req, InferRequest)

    it = iter(infer_queue)
    infer_request = next(it)
    assert isinstance(infer_request, InferRequest)
    assert infer_request.userdata is None
    with pytest.raises(StopIteration):
        next(it)


def test_infer_queue_userdata_is_empty(device):
    core = Core()
    param = ops.parameter([10])
    model = Model(ops.relu(param), [param])
    compiled_model = core.compile_model(model, device)
    infer_queue = AsyncInferQueue(compiled_model, 1)
    assert infer_queue.userdata == [None]


def test_infer_queue_userdata_is_empty_more_jobs(device):
    core = Core()
    param = ops.parameter([10])
    model = Model(ops.relu(param), [param])
    compiled_model = core.compile_model(model, device)
    infer_queue = AsyncInferQueue(compiled_model, 5)
    assert infer_queue.userdata == [None, None, None, None, None]


def test_infer_queue_fail_on_cpp_model(device):
    jobs = 6
    num_request = 4
    core = Core()
    model = get_relu_model()
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
    model = get_relu_model()
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


@skip_need_mock_op
@pytest.mark.parametrize("with_callback", [False, True])
def test_infer_queue_fail_in_inference(device, with_callback):
    jobs = 6
    num_request = 4
    core = Core()
    data = ops.parameter([10], dtype=np.float32, name="data")
    k_op = ops.parameter(Shape([]), dtype=np.int32, name="k")
    emb = ops.topk(data, k_op, axis=0, mode="max", sort="value")
    model = Model(emb, [data, k_op])
    compiled_model = core.compile_model(model, device)
    infer_queue = AsyncInferQueue(compiled_model, num_request)

    def callback(request, _):
        pytest.fail("Callback should not be called")

    if with_callback:
        infer_queue.set_callback(callback)

    data_tensor = Tensor(np.arange(10).astype(np.float32))
    k_tensor = Tensor(np.array(11, dtype=np.int32))

    with pytest.raises(RuntimeError) as e:
        for _ in range(jobs):
            infer_queue.start_async({"data": data_tensor, "k": k_tensor})
        infer_queue.wait_all()

    assert "Can not clone with new dims" in str(e.value)


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


@pytest.mark.parametrize("share_inputs", [True, False])
def test_results_async_infer(device, share_inputs):
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

    img = generate_image()
    infer_queue.set_callback(callback)
    for i in range(jobs):
        infer_queue.start_async({"data": img}, i, share_inputs=share_inputs)
    infer_queue.wait_all()

    request = compiled_model.create_infer_request()
    outputs = request.infer({0: img})

    for i in range(num_request):
        assert np.allclose(list(outputs.values()), list(
            infer_queue[i].results.values()))


@pytest.mark.parametrize("share_inputs", [True, False])
def test_array_like_input_async_infer_queue(device, share_inputs):
    class ArrayLikeObject:
        # Array-like object accepted by np.array to test inputs similar to torch tensor and tf.Tensor
        def __init__(self, array) -> None:
            self.data = array

        def __array__(self):
            return self.data

    jobs = 8
    ov_type = Type.f32
    input_shape = [2, 2]
    input_data = np.ascontiguousarray([[-2, -1], [0, 1]])
    param = ops.parameter(input_shape, ov_type)
    layer = ops.abs(param)
    model = Model([layer], [param])
    core = Core()
    compiled_model = core.compile_model(model, "CPU")

    model_input_object = ArrayLikeObject(input_data)
    model_input_list = [
        [ArrayLikeObject(deepcopy(input_data))] for _ in range(jobs)]

    # Test single array-like object in AsyncInferQueue.start_async()
    infer_queue_object = AsyncInferQueue(compiled_model, jobs)
    for _i in range(jobs):
        infer_queue_object.start_async(model_input_object)
    infer_queue_object.wait_all()

    for i in range(jobs):
        assert np.array_equal(
            infer_queue_object[i].get_output_tensor().data, np.abs(input_data))

    # Test list of array-like objects in AsyncInferQueue.start_async()
    infer_queue_list = AsyncInferQueue(compiled_model, jobs)
    for i in range(jobs):
        infer_queue_list.start_async(
            model_input_list[i], share_inputs=share_inputs)
    infer_queue_list.wait_all()

    for i in range(jobs):
        assert np.array_equal(
            infer_queue_list[i].get_output_tensor().data, np.abs(input_data))


@pytest.mark.parametrize("shared_flag", [True, False])
def test_shared_memory_deprecation(device, shared_flag):
    compiled, request, _, input_data = abs_model_with_data(
        device, Type.f32, np.float32)

    with pytest.warns(FutureWarning, match="`shared_memory` is deprecated and will be removed in 2024.0"):
        _ = compiled(input_data, shared_memory=shared_flag)

    with pytest.warns(FutureWarning, match="`shared_memory` is deprecated and will be removed in 2024.0"):
        _ = request.infer(input_data, shared_memory=shared_flag)

    with pytest.warns(FutureWarning, match="`shared_memory` is deprecated and will be removed in 2024.0"):
        request.start_async(input_data, shared_memory=shared_flag)
    request.wait()

    queue = AsyncInferQueue(compiled, jobs=1)

    with pytest.warns(FutureWarning, match="`shared_memory` is deprecated and will be removed in 2024.0"):
        queue.start_async(input_data, shared_memory=shared_flag)
    queue.wait_all()
