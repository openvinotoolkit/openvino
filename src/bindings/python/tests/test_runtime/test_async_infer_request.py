# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from copy import deepcopy
import numpy as np
import pytest
import time

import openvino.opset13 as ops
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
from tests.utils.helpers import (
    generate_image,
    get_relu_model,
    generate_concat_compiled_model_with_data,
    generate_abs_compiled_model_with_data,
)


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

        def __array__(self, dtype=None, copy=None):
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


@pytest.mark.skip(reason="Sporadically failed. Need further investigation. Ticket - 95967")
def test_cancel(device):
    core = Core()
    model = get_relu_model()
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


@pytest.mark.parametrize("share_inputs", [True, False])
def test_start_async(device, share_inputs):
    core = Core()
    model = get_relu_model()
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
        request.start_async({0: img}, share_inputs=share_inputs)
    for request in requests:
        request.wait()
        assert request.latency > 0
    assert callbacks_info["finished"] == jobs


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
def test_async_mixed_values(device, ov_type, numpy_dtype, share_inputs):
    request, tensor1, array1 = generate_concat_compiled_model_with_data(device=device, ov_type=ov_type, numpy_dtype=numpy_dtype)

    request.start_async([tensor1, array1], share_inputs=share_inputs)
    request.wait()
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
def test_async_single_input(device, ov_type, numpy_dtype, share_inputs):
    _, request, tensor1, array1 = generate_abs_compiled_model_with_data(device, ov_type, numpy_dtype)

    request.start_async(array1, share_inputs=share_inputs)
    request.wait()
    assert np.array_equal(request.get_output_tensor().data, np.abs(array1))

    request.start_async(tensor1, share_inputs=share_inputs)
    request.wait()
    assert np.array_equal(request.get_output_tensor().data, np.abs(tensor1.data))


@pytest.mark.parametrize("share_inputs", [True, False])
def test_array_like_input_async(device, share_inputs):
    class ArrayLikeObject:
        # Array-like object accepted by np.array to test inputs similar to torch tensor and tf.Tensor
        def __init__(self, array) -> None:
            self.data = array

        def __array__(self, dtype=None, copy=None):
            return np.array(self.data)

    _, request, _, input_data = generate_abs_compiled_model_with_data(device, Type.f32, np.single)
    model_input_object = ArrayLikeObject(input_data.tolist())
    model_input_list = [ArrayLikeObject(input_data.tolist())]
    # Test single array-like object in InferRequest().start_async()
    request.start_async(model_input_object, share_inputs=share_inputs)
    request.wait()
    assert np.array_equal(request.get_output_tensor().data, np.abs(input_data))

    # Test list of array-like objects in InferRequest().start_async()
    request.start_async(model_input_list)
    request.wait()
    assert np.array_equal(request.get_output_tensor().data, np.abs(input_data))
