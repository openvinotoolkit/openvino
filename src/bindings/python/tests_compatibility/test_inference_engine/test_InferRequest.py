# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os
import pytest
import threading
from datetime import datetime
import time

from openvino.inference_engine import ie_api as ie
from tests_compatibility.conftest import model_path, create_encoder
from tests_compatibility.test_utils.test_utils import generate_image, generate_relu_model
import ngraph as ng
from ngraph.impl import Function, Type

test_net_xml, test_net_bin = model_path()


def create_function_with_memory(input_shape, data_type):
    input_data = ng.parameter(input_shape, name="input_data", dtype=data_type)
    rv = ng.read_value(input_data, "var_id_667")
    add = ng.add(rv, input_data, name="MemoryAdd")
    node = ng.assign(add, "var_id_667")
    res = ng.result(add, "res")
    func = Function(results=[res], sinks=[node], parameters=[input_data], name="name")
    caps = Function.to_capsule(func)
    return caps


def load_sample_model(device, num_requests=1):
    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    executable_network = ie_core.load_network(net, device, num_requests=num_requests)
    return executable_network


def test_input_blobs(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    executable_network = ie_core.load_network(net, device, num_requests=2)
    td = ie.TensorDesc("FP32", (1, 3, 32, 32), "NCHW")
    assert executable_network.requests[0].input_blobs['data'].tensor_desc == td


def test_output_blobs(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    executable_network = ie_core.load_network(net, device, num_requests=2)
    td = ie.TensorDesc("FP32", (1, 10), "NC")
    assert executable_network.requests[0].output_blobs['fc_out'].tensor_desc == td


def test_inputs_list(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    executable_network = ie_core.load_network(net, device, num_requests=2)

    for req in executable_network.requests:
        assert len(req._inputs_list) == 1
        assert "data" in req._inputs_list
    del ie_core


def test_outputs_list(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    executable_network = ie_core.load_network(net, device, num_requests=2)

    for req in executable_network.requests:
        assert len(req._outputs_list) == 1
        assert "fc_out" in req._outputs_list
    del ie_core


def test_access_input_buffer(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    executable_network = ie_core.load_network(net, device, num_requests=1)
    buffer = executable_network.requests[0]._get_blob_buffer("data".encode()).to_numpy()
    assert buffer.shape == (1, 3, 32, 32)
    assert buffer.strides == (12288, 4096, 128, 4)
    assert buffer.dtype == np.float32
    del executable_network
    del ie_core
    del net


def test_access_output_buffer(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    executable_network = ie_core.load_network(net, device, num_requests=1)
    buffer = executable_network.requests[0]._get_blob_buffer("fc_out".encode()).to_numpy()
    assert buffer.shape == (1, 10)
    assert buffer.strides == (40, 4)
    assert buffer.dtype == np.float32
    del executable_network
    del ie_core
    del net


def test_write_to_input_blobs_directly(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    executable_network = ie_core.load_network(net, device, num_requests=1)
    img = generate_image()
    request = executable_network.requests[0]
    input_data = request.input_blobs["data"]
    input_data.buffer[:] = img
    assert np.array_equal(executable_network.requests[0].input_blobs["data"].buffer, img)
    del executable_network
    del ie_core
    del net


def test_write_to_input_blobs_copy(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    executable_network = ie_core.load_network(net, device, num_requests=1)
    img = generate_image()
    request = executable_network.requests[0]
    request.input_blobs["data"].buffer[:] = img
    assert np.allclose(executable_network.requests[0].input_blobs["data"].buffer, img)
    del executable_network
    del ie_core
    del net


def test_infer(device):
    ie_core = ie.IECore()
    net = generate_relu_model([1, 3, 32, 32])
    exec_net = ie_core.load_network(net, device, num_requests=1)
    img = generate_image()
    request = exec_net.requests[0]
    request.infer({'parameter': img})
    res = request.output_blobs['relu'].buffer
    assert np.argmax(res) == 531
    del exec_net
    del ie_core
    del net


def test_async_infer_default_timeout(device):
    ie_core = ie.IECore()
    net = generate_relu_model([1, 3, 32, 32])
    exec_net = ie_core.load_network(net, device, num_requests=1)
    img = generate_image()
    request = exec_net.requests[0]
    request.async_infer({'parameter': img})
    request.wait()
    res = request.output_blobs['relu'].buffer
    assert np.argmax(res) == 531
    del exec_net
    del ie_core
    del net


def test_async_infer_wait_finish(device):
    ie_core = ie.IECore()
    net = generate_relu_model([1, 3, 32, 32])
    exec_net = ie_core.load_network(net, device, num_requests=1)
    img = generate_image()
    request = exec_net.requests[0]
    request.async_infer({'parameter': img})
    request.wait(ie.WaitMode.RESULT_READY)
    res = request.output_blobs['relu'].buffer
    assert np.argmax(res) == 531
    del exec_net
    del ie_core
    del net


def test_async_infer_wait_time(device):
    ie_core = ie.IECore()
    net = generate_relu_model([1, 3, 32, 32])
    exec_net = ie_core.load_network(net, device, num_requests=2)
    img = generate_image()
    request = exec_net.requests[0]
    request.async_infer({'parameter': img})
    start_time = datetime.utcnow()
    status = request.wait(ie.WaitMode.RESULT_READY)
    assert status == ie.StatusCode.OK
    time_delta = datetime.utcnow() - start_time
    latency_ms = (time_delta.microseconds / 1000) + (time_delta.seconds * 1000)
    timeout = max(100, latency_ms)
    request = exec_net.requests[1]
    request.async_infer({'parameter': img})
    max_repeat = 10
    status = ie.StatusCode.REQUEST_BUSY
    i = 0
    while i < max_repeat and status != ie.StatusCode.OK:
        status = request.wait(timeout)
        i += 1
    assert status == ie.StatusCode.OK
    res = request.output_blobs['relu'].buffer
    assert np.argmax(res) == 531
    del exec_net
    del ie_core
    del net


def test_async_infer_wait_status(device):
    ie_core = ie.IECore()
    net = generate_relu_model([1, 3, 32, 32])
    exec_net = ie_core.load_network(net, device, num_requests=1)
    img = generate_image()
    request = exec_net.requests[0]
    request.async_infer({'parameter': img})
    request.wait(ie.WaitMode.RESULT_READY)
    res = request.output_blobs['relu'].buffer
    assert np.argmax(res) == 531
    status = request.wait(ie.WaitMode.STATUS_ONLY)
    assert status == ie.StatusCode.OK
    del exec_net
    del ie_core
    del net


def test_async_infer_fill_inputs(device):
    ie_core = ie.IECore()
    net = generate_relu_model([1, 3, 32, 32])
    exec_net = ie_core.load_network(net, device, num_requests=1)
    img = generate_image()
    request = exec_net.requests[0]
    request.input_blobs['parameter'].buffer[:] = img
    request.async_infer()
    status_end = request.wait()
    assert status_end == ie.StatusCode.OK
    res = request.output_blobs['relu'].buffer
    assert np.argmax(res[0]) == 531
    del exec_net
    del ie_core
    del net


def test_infer_modify_outputs(device):
    ie_core = ie.IECore()
    net = generate_relu_model([1, 3, 32, 32])
    exec_net = ie_core.load_network(net, device, num_requests=1)
    img = generate_image()
    request = exec_net.requests[0]
    outputs0 = exec_net.infer({'parameter': img})
    status_end = request.wait()
    assert status_end == ie.StatusCode.OK
    assert np.argmax(outputs0['relu']) == 531
    outputs0['relu'][:] = np.zeros(shape=(1, 3, 32, 32), dtype=np.float32)
    outputs1 = request.output_blobs
    assert np.argmax(outputs1['relu'].buffer) == 531
    outputs1['relu'].buffer[:] = np.ones(shape=(1, 3, 32, 32), dtype=np.float32)
    outputs2 = request.output_blobs
    assert np.argmax(outputs2['relu'].buffer) == 531
    del exec_net
    del ie_core
    del net


def test_async_infer_callback(device):
    def static_vars(**kwargs):
        def decorate(func):
            for k in kwargs:
                setattr(func, k, kwargs[k])
            return func

        return decorate

    @static_vars(callback_called=0)
    def callback(self, status):
        callback.callback_called = 1

    ie_core = ie.IECore()
    net = generate_relu_model([1, 3, 32, 32])
    exec_net = ie_core.load_network(net, device, num_requests=1)
    img = generate_image()
    request = exec_net.requests[0]
    request.set_completion_callback(callback)
    request.async_infer({'parameter': img})
    status = request.wait()
    assert status == ie.StatusCode.OK
    res = request.output_blobs['relu'].buffer
    assert np.argmax(res) == 531
    assert callback.callback_called == 1
    del exec_net
    del ie_core


def test_async_infer_callback_wait_before_start(device):
    def static_vars(**kwargs):
        def decorate(func):
            for k in kwargs:
                setattr(func, k, kwargs[k])
            return func
        return decorate

    @static_vars(callback_called=0)
    def callback(self, status):
        callback.callback_called = 1

    ie_core = ie.IECore()
    net = generate_relu_model([1, 3, 32, 32])
    exec_net = ie_core.load_network(net, device, num_requests=1)
    img = generate_image()
    request = exec_net.requests[0]
    request.set_completion_callback(callback)
    status = request.wait()
    # Plugin API 2.0 has the different behavior will not return this status
    # assert status == ie.StatusCode.INFER_NOT_STARTED
    request.async_infer({'parameter': img})
    status = request.wait()
    assert status == ie.StatusCode.OK
    res = request.output_blobs['relu'].buffer
    assert np.argmax(res) == 531
    assert callback.callback_called == 1
    del exec_net
    del ie_core


def test_async_infer_callback_wait_in_callback(device):
    class InferReqWrap:
        def __init__(self, request):
            self.request = request
            self.cv = threading.Condition()
            self.request.set_completion_callback(self.callback)
            self.status_code = self.request.wait(ie.WaitMode.STATUS_ONLY)
            # Plugin API 2.0 has the different behavior will not return this status
            # assert self.status_code == ie.StatusCode.INFER_NOT_STARTED

        def callback(self, statusCode, userdata):
            self.status_code = self.request.wait(ie.WaitMode.STATUS_ONLY)
            self.cv.acquire()
            self.cv.notify()
            self.cv.release()

        def execute(self, input_data):
            self.request.async_infer(input_data)
            self.cv.acquire()
            self.cv.wait()
            self.cv.release()
            status = self.request.wait(ie.WaitMode.RESULT_READY)
            assert status == ie.StatusCode.OK
            assert self.status_code == ie.StatusCode.RESULT_NOT_READY

    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    exec_net = ie_core.load_network(net, device, num_requests=1)
    img = generate_image()
    request_wrap = InferReqWrap(exec_net.requests[0])
    request_wrap.execute({'data': img})
    del exec_net
    del ie_core


def test_async_infer_wait_while_callback_will_not_finish(device):
    def callback(status, callback_status):
        time.sleep(0.01)
        callback_status['finished'] = True

    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    exec_net = ie_core.load_network(net, device, num_requests=1)
    callback_status = {}
    callback_status['finished'] = False
    request = exec_net.requests[0]
    request.set_completion_callback(callback, py_data=callback_status)
    img = generate_image()
    request.async_infer({'data': img})
    request.wait()
    assert callback_status['finished'] == True


def test_get_perf_counts(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    ie_core.set_config({"PERF_COUNT": "YES"}, device)
    exec_net = ie_core.load_network(net, device)
    img = generate_image()
    request = exec_net.requests[0]
    request.infer({'data': img})
    pc = request.get_perf_counts()
    assert pc['29/WithoutBiases']["status"] == "EXECUTED"
    del exec_net
    del ie_core
    del net


def test_blob_setter(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    exec_net_1 = ie_core.load_network(network=net, device_name=device, num_requests=1)

    net.input_info['data'].layout = "NHWC"
    exec_net_2 = ie_core.load_network(network=net, device_name=device, num_requests=1)

    img = generate_image()
    res_1 = np.sort(exec_net_1.infer({"data": img})['fc_out'])

    img = np.transpose(img, axes=(0, 2, 3, 1)).astype(np.float32)
    tensor_desc = ie.TensorDesc("FP32", [1, 3, 32, 32], "NHWC")
    img_blob = ie.Blob(tensor_desc, img)
    request = exec_net_2.requests[0]
    request.set_blob('data', img_blob)
    request.infer()
    res_2 = np.sort(request.output_blobs['fc_out'].buffer)
    assert np.allclose(res_1, res_2, atol=1e-2, rtol=1e-2)


def test_getting_preprocess(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    exec_net = ie_core.load_network(network=net, device_name=device, num_requests=1)
    request = exec_net.requests[0]
    preprocess_info = request.preprocess_info["data"]
    assert isinstance(preprocess_info, ie.PreProcessInfo)
    assert preprocess_info.mean_variant == ie.MeanVariant.NONE


def test_resize_algorithm_work(device):
    ie_core = ie.IECore()
    net = generate_relu_model([1, 3, 32, 32])
    exec_net_1 = ie_core.load_network(network=net, device_name=device, num_requests=1)

    img = generate_image()
    res_1 = np.sort(exec_net_1.infer({"parameter": img})['relu'])

    net.input_info['parameter'].preprocess_info.resize_algorithm = ie.ResizeAlgorithm.RESIZE_BILINEAR

    exec_net_2 = ie_core.load_network(net, device)

    tensor_desc = ie.TensorDesc("FP32", [1, 3, img.shape[2], img.shape[3]], "NCHW")
    img_blob = ie.Blob(tensor_desc, img)
    request = exec_net_2.requests[0]
    assert request.preprocess_info["parameter"].resize_algorithm == ie.ResizeAlgorithm.RESIZE_BILINEAR
    request.set_blob('parameter', img_blob)
    request.infer()
    res_2 = np.sort(request.output_blobs['relu'].buffer)

    assert np.allclose(res_1, res_2, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("mode", ["set_init_memory_state", "reset_memory_state", "normal"])
@pytest.mark.parametrize("data_type", ["FP32", "FP16", "I32"])
@pytest.mark.parametrize("input_shape", [[10], [10, 10], [10, 10, 10], [2, 10, 10, 10]])
@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU",
                    reason=f"Can't run test on device {os.environ.get('TEST_DEVICE', 'CPU')}, "
                    "Memory layers fully supported only on CPU")
def test_query_state_write_buffer(device, input_shape, data_type, mode):
    ie_core = ie.IECore()

    layout = ["C", "HW", "CHW", "NCHW"]

    from openvino.inference_engine import TensorDesc, Blob, format_map

    net = ie.IENetwork(create_function_with_memory(input_shape, format_map[data_type]))
    ie_core = ie.IECore()
    exec_net = ie_core.load_network(network=net, device_name=device, num_requests=1)
    request = exec_net.requests[0]
    mem_states = request.query_state()
    mem_state = mem_states[0]

    assert mem_state.name == 'var_id_667'
    # todo: Uncomment after fix 45611,
    #  CPU plugin returns outputs and memory state in FP32 in case of FP16 original precision
    #assert mem_state.state.tensor_desc.precision == data_type

    for i in range(1, 10):
        if mode == "set_init_memory_state":
            # create initial value
            const_init = 5
            init_array = np.full(input_shape, const_init, dtype=format_map[mem_state.state.tensor_desc.precision])
            tensor_desc = TensorDesc(mem_state.state.tensor_desc.precision, input_shape, layout[len(input_shape) - 1])
            blob = Blob(tensor_desc, init_array)
            mem_state.state = blob

            res = exec_net.infer({"input_data": np.full(input_shape, 1, dtype=format_map[data_type])})
            expected_res = np.full(input_shape, 1 + const_init, dtype=format_map[data_type])
        elif mode == "reset_memory_state":
            # reset initial state of ReadValue to zero
            mem_state.reset()
            res = exec_net.infer({"input_data": np.full(input_shape, 1, dtype=format_map[data_type])})

            # always ones
            expected_res = np.full(input_shape, 1, dtype=format_map[data_type])
        else:
            res = exec_net.infer({"input_data": np.full(input_shape, 1, dtype=format_map[data_type])})
            expected_res = np.full(input_shape, i, dtype=format_map[data_type])

        assert np.allclose(res['MemoryAdd'], expected_res, atol=1e-6), \
            "Expected values: {} \n Actual values: {} \n".format(expected_res, res)


def test_set_blob_with_incorrect_name(device):
    function = create_encoder([4, 4, 20, 20])
    net = ng.function_to_cnn(function)
    ie_core = ie.IECore()
    exec_net = ie_core.load_network(net, device)
    tensor_desc = exec_net.requests[0].input_blobs["data"].tensor_desc
    tensor_desc.dims = [4, 4, 20, 20]
    blob = ie.Blob(tensor_desc)
    with pytest.raises(RuntimeError) as e:
        exec_net.requests[0].set_blob("incorrect_name", blob)
    assert f"Failed to find input or output with name: 'incorrect_name'" in str(e.value)


def test_set_blob_with_incorrect_size(device):
    function = create_encoder([4, 4, 20, 20])
    net = ng.function_to_cnn(function)
    ie_core = ie.IECore()
    exec_net = ie_core.load_network(net, device)
    tensor_desc = exec_net.requests[0].input_blobs["data"].tensor_desc
    tensor_desc.dims = [tensor_desc.dims[0]*2, 4, 20, 20]
    blob = ie.Blob(tensor_desc)
    with pytest.raises(RuntimeError) as e:
        exec_net.requests[0].set_blob("data", blob)
    assert f"Can't set the input tensor" in str(e.value)
    with pytest.raises(RuntimeError) as e:
        exec_net.requests[0].set_blob("out", blob)
    assert f"Can't set the output tensor" in str(e.value)
