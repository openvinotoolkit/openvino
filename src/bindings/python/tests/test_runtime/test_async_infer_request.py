# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from copy import deepcopy
import gc
import sys
import threading
import weakref
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


@pytest.mark.skip(reason="CVS-189144")
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
        assert np.allclose(list(outputs.values()), list(infer_queue[i].results.values()))


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
    request, tensor1, array1 = generate_concat_compiled_model_with_data(
        device=device, ov_type=ov_type, numpy_dtype=numpy_dtype
    )

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


@pytest.mark.skipif(sys.platform != "linux", reason="relies on glibc LIFO freelist address reuse")
def test_infer_queue_share_inputs_data_integrity(device):
    # Regression test for output data corruption caused by use-after-free when share_inputs=True.
    jobs = 8
    num_request = 4
    core = Core()
    model = get_relu_model()
    compiled_model = core.compile_model(model, device)
    infer_queue = AsyncInferQueue(compiled_model, num_request)

    outputs = [None] * jobs

    def callback(request, job_id):
        outputs[job_id] = request.get_output_tensor(0).data.copy()

    infer_queue.set_callback(callback)

    shape = (1, 3, 32, 32)
    fillers = []  # kept alive so the poison buffer is not freed before inference

    for i in range(jobs):
        # Unique positive fill value per job; relu(positive) = positive.
        img = np.full(shape, float(i + 1), dtype=np.float32)
        infer_queue.start_async({"data": img}, i, share_inputs=True)
        # Drop the caller's reference and trigger GC so the allocator reclaims the buffer immediately.
        del img
        gc.collect()
        # Allocate a same-sized array with negative values.
        # On CPython this lands at the exact same address the allocator just freed (100% reproducible).
        fillers.append(np.full(shape, float(-(i + 1)), dtype=np.float32))

    infer_queue.wait_all()

    for i in range(jobs):
        expected = float(i + 1)
        actual = float(outputs[i].flat[0])
        assert abs(actual - expected) < 0.01, (
            f"Job {i}: output corrupted by use-after-free in shared inputs. "
            f"Expected {expected} (relu of original input), got {actual}. "
        )


@pytest.mark.skipif(sys.platform != "linux", reason="relies on glibc LIFO freelist address reuse")
def test_infer_queue_share_inputs_hang(device):
    # Hang-simulation regression test for the use-after-free with share_inputs=True.
    shape = (1, 3, 32, 32)
    expected = 7.0
    # 3s is far more than enough for a tiny relu to complete
    # if the event is not set within this window the callback already fired with wrong output.
    hang_timeout = 3.0

    core = Core()
    model = get_relu_model(list(shape))
    compiled_model = core.compile_model(model, device)
    infer_queue = AsyncInferQueue(compiled_model, 1)

    correct_output_seen = threading.Event()

    def callback(request, _):
        value = float(request.get_output_tensor(0).data.flat[0])
        if abs(value - expected) < 0.1:
            correct_output_seen.set()

    infer_queue.set_callback(callback)

    img = np.full(shape, expected, dtype=np.float32)
    infer_queue.start_async({"data": img}, None, share_inputs=True)

    # Drop the caller-side reference and force GC, so the allocator reclaims the buffer.
    # On CPython the very next same-sized allocation (below) reuses the same address deterministically.
    del img
    gc.collect()

    # Zero-filled array lands at the freed address (same glibc free-list slot).
    # relu(0.0) = 0.0 ≠ EXPECTED — if inference reads this the callback will NOT signal the event.
    poison = np.zeros(shape, dtype=np.float32)  # noqa: F841 – must stay alive

    event_was_set = correct_output_seen.wait(timeout=hang_timeout)
    infer_queue.wait_all()

    assert event_was_set, (
        f"HANG SIMULATED (timed out after {hang_timeout}s): "
        f"callback never received expected output {expected}. "
        "The shared input buffer was freed and zeroed before inference read it. "
    )


def test_infer_queue_share_inputs_array_lifetime(device):
    # Regression test for use-after-free when share_inputs=True.
    #
    # Platform-independent: uses a weakref to verify that the queue holds a Python reference to each numpy array
    # for the lifetime of the corresponding async inference request.
    jobs = 8
    num_request = 4
    core = Core()
    model = get_relu_model()
    compiled_model = core.compile_model(model, device)
    infer_queue = AsyncInferQueue(compiled_model, num_request)
    infer_queue.set_callback(lambda request, userdata: None)

    for i in range(jobs):
        img = generate_image()
        wr = weakref.ref(img)
        infer_queue.start_async({"data": img}, i, share_inputs=True)
        del img
        gc.collect()
        assert wr() is not None, (
            f"Input numpy array for job {i} was freed while async inference "
            "may still be using its memory (use-after-free). "
            "The queue must keep shared input arrays alive via _inputs_data[handle]."
        )

    infer_queue.wait_all()


def test_infer_queue_inputs_data_not_created_when_always_non_shared(device):
    # _inputs_data must not be created at all when share_inputs is always False.
    # Verifies that the hasattr guard does not fabricate the dict on non-shared paths.
    core = Core()
    model = get_relu_model()
    compiled_model = core.compile_model(model, device)
    infer_queue = AsyncInferQueue(compiled_model, 1)
    infer_queue.set_callback(lambda request, userdata: None)

    for _ in range(3):
        infer_queue.start_async({"data": generate_image()}, None, share_inputs=False)
        infer_queue.wait_all()

    assert not hasattr(infer_queue, "_inputs_data"), (
        "_inputs_data must not be created when share_inputs is always False — "
        "the dict should only be allocated on the first share_inputs=True call."
    )


def test_infer_queue_non_shared_then_shared_establishes_ownership(device):
    # A handle whose first use is share_inputs=False must correctly switch to
    # owning the array when subsequently called with share_inputs=True.
    core = Core()
    model = get_relu_model()
    compiled_model = core.compile_model(model, device)
    infer_queue = AsyncInferQueue(compiled_model, 1)
    infer_queue.set_callback(lambda request, userdata: None)

    # First use: share_inputs=False — _inputs_data must NOT be created.
    infer_queue.start_async({"data": generate_image()}, None, share_inputs=False)
    infer_queue.wait_all()
    assert not hasattr(infer_queue, "_inputs_data"), (
        "No _inputs_data should exist after a non-shared first call."
    )

    # Second use: share_inputs=True — _inputs_data must now be created and hold the array.
    img = generate_image()
    wr = weakref.ref(img)
    infer_queue.start_async({"data": img}, None, share_inputs=True)
    infer_queue.wait_all()
    del img
    gc.collect()

    assert hasattr(infer_queue, "_inputs_data"), (
        "_inputs_data must be created on the first share_inputs=True call."
    )
    assert wr() is not None, (
        "Array must be alive — queue must hold it via _inputs_data after share_inputs=True."
    )


def test_infer_queue_shared_to_shared_releases_old_array(device):
    # When the same handle is reused with share_inputs=True, the OLD array must be
    # released: _inputs_data[handle] is overwritten with the new array reference.
    core = Core()
    model = get_relu_model()
    compiled_model = core.compile_model(model, device)
    infer_queue = AsyncInferQueue(compiled_model, 1)
    infer_queue.set_callback(lambda request, userdata: None)

    img1 = generate_image()
    wr1 = weakref.ref(img1)
    infer_queue.start_async({"data": img1}, None, share_inputs=True)
    infer_queue.wait_all()
    del img1
    gc.collect()
    assert wr1() is not None  # queue holds it

    img2 = generate_image()
    wr2 = weakref.ref(img2)
    infer_queue.start_async({"data": img2}, None, share_inputs=True)
    infer_queue.wait_all()
    del img2
    gc.collect()

    assert wr1() is None, (
        "Old shared array must be released when the handle is reused with share_inputs=True: "
        "_inputs_data[handle] must be overwritten, dropping the old reference."
    )
    assert wr2() is not None, "New shared array must be alive — queue holds it."


def test_infer_queue_shared_buffer_kept_alive_during_non_shared_reuse(device):
    # When share_inputs=False follows share_inputs=True on the same handle, the existing
    # ov::Tensor still wraps the original numpy buffer.  _data_dispatch (non-shared path)
    # copies new data INTO that same buffer via tensor.data[:] = inputs[:], so the OLD
    # numpy array must remain alive for the whole lifetime of the tensor.
    #
    # The old array is only released when the NEXT share_inputs=True call replaces the
    # tensor with a fresh zero-copy wrapper around a different numpy buffer.
    core = Core()
    model = get_relu_model()
    compiled_model = core.compile_model(model, device)
    infer_queue = AsyncInferQueue(compiled_model, 1)
    infer_queue.set_callback(lambda request, userdata: None)

    # Step 1: shared — tensor wraps img1's buffer; queue holds img1 in _inputs_data[0].
    img1 = generate_image()
    wr1 = weakref.ref(img1)
    infer_queue.start_async({"data": img1}, None, share_inputs=True)
    infer_queue.wait_all()
    del img1
    gc.collect()
    assert wr1() is not None, "Queue must hold img1 via _inputs_data after share_inputs=True."

    # Step 2: non-shared — copies img2's data INTO img1's buffer; tensor unchanged.
    # img1 MUST NOT be freed here: the tensor's data pointer still points to its buffer.
    infer_queue.start_async({"data": generate_image()}, None, share_inputs=False)
    infer_queue.wait_all()
    gc.collect()
    assert wr1() is not None, (
        "img1 must still be alive after a non-shared reuse: update_tensor copies data "
        "into the existing tensor buffer (which wraps img1's memory). Releasing img1 here "
        "would create a dangling pointer and cause a use-after-free during inference."
    )

    # Step 3: another shared call — new zero-copy tensor created; _inputs_data[0] updated;
    # img1 reference dropped → img1 finally released.
    img3 = generate_image()
    wr3 = weakref.ref(img3)
    infer_queue.start_async({"data": img3}, None, share_inputs=True)
    infer_queue.wait_all()
    del img3
    gc.collect()
    assert wr1() is None, (
        "img1 must be released once share_inputs=True replaces the tensor and overwrites "
        "_inputs_data[0] with the new array reference."
    )
    assert wr3() is not None, "New shared array must be alive — queue holds it."


def test_infer_queue_multi_handle_inputs_data_independence(device):
    # _inputs_data[A] and _inputs_data[B] must be managed independently.
    # Updating or releasing one handle's entry must not affect the other.
    core = Core()
    model = get_relu_model()
    compiled_model = core.compile_model(model, device)
    infer_queue = AsyncInferQueue(compiled_model, 2)
    infer_queue.set_callback(lambda request, userdata: None)

    # Fill both handles with share_inputs=True.
    img_a = generate_image()
    wr_a = weakref.ref(img_a)
    infer_queue.start_async({"data": img_a}, None, share_inputs=True)

    img_b = generate_image()
    wr_b = weakref.ref(img_b)
    infer_queue.start_async({"data": img_b}, None, share_inputs=True)

    infer_queue.wait_all()
    del img_a, img_b
    gc.collect()
    assert wr_a() is not None and wr_b() is not None, (
        "Both arrays must be alive — each handle independently owns its _inputs_data entry."
    )

    # Reuse ONE handle with a fresh share_inputs=True call.
    # The OTHER handle's entry must remain untouched.
    img_new = generate_image()
    wr_new = weakref.ref(img_new)
    infer_queue.start_async({"data": img_new}, None, share_inputs=True)
    infer_queue.wait_all()
    del img_new
    gc.collect()

    # Exactly one of the original arrays must have been released (the reused handle's).
    dead = sum(1 for wr in (wr_a, wr_b) if wr() is None)
    alive = sum(1 for wr in (wr_a, wr_b) if wr() is not None)
    assert dead == 1, (
        f"Exactly one original array must be released (the reused handle's). "
        f"Got dead={dead}, alive={alive}."
    )
    assert alive == 1, "The other original array must still be alive — its handle was not touched."
    assert wr_new() is not None, "New shared array must be alive — queue holds it."


def test_infer_queue_many_shared_cycles_no_use_after_free(device):
    # Stress: many consecutive share_inputs=True calls on the same 1-slot queue.
    # After each call the current array must be alive; the previous one must be dead.
    core = Core()
    model = get_relu_model()
    compiled_model = core.compile_model(model, device)
    infer_queue = AsyncInferQueue(compiled_model, 1)
    infer_queue.set_callback(lambda request, userdata: None)

    prev_wr = None
    for cycle in range(6):
        img = generate_image()
        wr = weakref.ref(img)
        infer_queue.start_async({"data": img}, None, share_inputs=True)
        infer_queue.wait_all()
        del img
        gc.collect()

        assert wr() is not None, (
            f"Cycle {cycle}: current shared array must be alive — queue owns the reference."
        )
        if prev_wr is not None:
            assert prev_wr() is None, (
                f"Cycle {cycle}: previous shared array must be released — "
                "_inputs_data[handle] must be overwritten when the handle is reused."
            )
        prev_wr = wr
