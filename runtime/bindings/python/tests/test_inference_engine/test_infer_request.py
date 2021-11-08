# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os
import pytest
import datetime

from ..conftest import image_path, model_path
from openvino import Core, Tensor, ProfilingInfo

is_myriad = os.environ.get("TEST_DEVICE") == "MYRIAD"
test_net_xml, test_net_bin = model_path(is_myriad)


def read_image():
    import cv2
    n, c, h, w = (1, 3, 32, 32)
    image = cv2.imread(image_path())
    if image is None:
        raise FileNotFoundError("Input image not found")

    image = cv2.resize(image, (h, w)) / 255
    image = image.transpose((2, 0, 1)).astype(np.float32)
    image = image.reshape((n, c, h, w))
    return image


def test_get_profiling_info(device):
    core = Core()
    func = core.read_model(test_net_xml, test_net_bin)
    core.set_config({"PERF_COUNT": "YES"}, device)
    exec_net = core.compile_model(func, device)
    img = read_image()
    request = exec_net.create_infer_request()
    request.infer({0: img})
    prof_info = request.get_profiling_info()
    soft_max_node = next(node for node in prof_info if node.node_name == "fc_out")
    assert soft_max_node.node_type == "Softmax"
    assert soft_max_node.status == ProfilingInfo.Status.OPTIMIZED_OUT
    assert isinstance(soft_max_node.real_time, datetime.timedelta)
    assert isinstance(soft_max_node.cpu_time, datetime.timedelta)
    assert isinstance(soft_max_node.exec_type, str)


def test_tensor_setter(device):
    core = Core()
    func = core.read_model(test_net_xml, test_net_bin)
    exec_net_1 = core.compile_model(network=func, device_name=device)
    exec_net_2 = core.compile_model(network=func, device_name=device)

    img = read_image()
    tensor = Tensor(img)

    request1 = exec_net_1.create_infer_request()
    request1.set_tensor("data", tensor)
    t1 = request1.get_tensor("data")

    assert np.allclose(tensor.data, t1.data, atol=1e-2, rtol=1e-2)

    res = request1.infer({0: tensor})
    res_1 = np.sort(res[0])
    t2 = request1.get_tensor("fc_out")
    assert np.allclose(t2.data, res[0].data, atol=1e-2, rtol=1e-2)

    request = exec_net_2.create_infer_request()
    res = request.infer({"data": tensor})
    res_2 = np.sort(request.get_tensor("fc_out").data)
    assert np.allclose(res_1, res_2, atol=1e-2, rtol=1e-2)

    request.set_tensor("data", tensor)
    t3 = request.get_tensor("data")
    assert np.allclose(t3.data, t1.data, atol=1e-2, rtol=1e-2)


def test_set_tensors(device):
    core = Core()
    func = core.read_model(test_net_xml, test_net_bin)
    exec_net = core.compile_model(func, device)

    data1 = read_image()
    tensor1 = Tensor(data1)
    data2 = np.ones(shape=(1, 10), dtype=np.float32)
    tensor2 = Tensor(data2)
    data3 = np.ones(shape=(1, 3, 32, 32), dtype=np.float32)
    tensor3 = Tensor(data3)
    data4 = np.zeros(shape=(1, 10), dtype=np.float32)
    tensor4 = Tensor(data4)

    request = exec_net.create_infer_request()
    request.set_tensors({"data": tensor1, "fc_out": tensor2})
    t1 = request.get_tensor("data")
    t2 = request.get_tensor("fc_out")
    assert np.allclose(tensor1.data, t1.data, atol=1e-2, rtol=1e-2)
    assert np.allclose(tensor2.data, t2.data, atol=1e-2, rtol=1e-2)

    request.set_output_tensors({0: tensor2})
    output_node = exec_net.outputs[0]
    t3 = request.get_tensor(output_node)
    assert np.allclose(tensor2.data, t3.data, atol=1e-2, rtol=1e-2)

    request.set_input_tensors({0: tensor1})
    output_node = exec_net.inputs[0]
    t4 = request.get_tensor(output_node)
    assert np.allclose(tensor1.data, t4.data, atol=1e-2, rtol=1e-2)

    output_node = exec_net.inputs[0]
    request.set_tensor(output_node, tensor3)
    t5 = request.get_tensor(output_node)
    assert np.allclose(tensor3.data, t5.data, atol=1e-2, rtol=1e-2)

    request.set_input_tensor(tensor3)
    t6 = request.get_tensor(request.inputs[0])
    assert np.allclose(tensor3.data, t6.data, atol=1e-2, rtol=1e-2)

    request.set_input_tensor(0, tensor1)
    t7 = request.get_tensor(request.inputs[0])
    assert np.allclose(tensor1.data, t7.data, atol=1e-2, rtol=1e-2)

    request.set_output_tensor(tensor2)
    t8 = request.get_tensor(request.outputs[0])
    assert np.allclose(tensor2.data, t8.data, atol=1e-2, rtol=1e-2)

    request.set_output_tensor(0, tensor4)
    t9 = request.get_tensor(request.outputs[0])
    assert np.allclose(tensor4.data, t9.data, atol=1e-2, rtol=1e-2)


def test_cancel(device):
    core = Core()
    func = core.read_model(test_net_xml, test_net_bin)
    exec_net = core.compile_model(func, device)
    img = read_image()
    request = exec_net.create_infer_request()

    def callback(e):
        raise Exception(e)

    request.set_callback(callback)
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


def test_infer_mixed_keys(device):
    core = Core()
    func = core.read_model(test_net_xml, test_net_bin)
    core.set_config({"PERF_COUNT": "YES"}, device)
    exec_net = core.compile_model(func, device)

    img = read_image()
    tensor = Tensor(img)

    data2 = np.ones(shape=(1, 10), dtype=np.float32)
    tensor2 = Tensor(data2)

    request = exec_net.create_infer_request()
    with pytest.raises(TypeError) as e:
        request.infer({0: tensor, "fc_out": tensor2})
    assert "incompatible function arguments!" in str(e.value)
