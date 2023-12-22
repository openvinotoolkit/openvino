# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os
import pytest
import time

from openvino.inference_engine import ie_api as ie
from tests_compatibility.conftest import model_path
from tests_compatibility.test_utils.test_utils import generate_image, generate_relu_model


test_net_xml, test_net_bin = model_path(False)


def test_infer(device):
    ie_core = ie.IECore()
    net = generate_relu_model([1, 3, 32, 32])
    exec_net = ie_core.load_network(net, device)
    img = generate_image()
    res = exec_net.infer({'parameter': img})
    assert np.argmax(res['relu'][0]) == 531
    del exec_net
    del ie_core


def test_infer_wrong_input_name(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie_core.load_network(net, device)
    img = generate_image()
    with pytest.raises(AssertionError) as e:
        exec_net.infer({'_data_': img})
    assert "No input with name _data_ found in network" in str(e.value)
    del exec_net
    del ie_core


def test_input_info(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie_core.load_network(net, device, num_requests=5)
    assert isinstance(exec_net.input_info['data'], ie.InputInfoCPtr)
    assert exec_net.input_info['data'].name == "data"
    assert exec_net.input_info['data'].precision == "FP32"
    assert isinstance(exec_net.input_info['data'].input_data, ie.DataPtr)
    del exec_net
    del ie_core


def test_outputs(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie_core.load_network(net, device, num_requests=5)
    assert len(exec_net.outputs) == 1
    assert "fc_out" in exec_net.outputs
    assert isinstance(exec_net.outputs['fc_out'], ie.CDataPtr)
    del exec_net
    del ie_core


def test_access_requests(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie_core.load_network(net, device, num_requests=5)
    assert len(exec_net.requests) == 5
    assert isinstance(exec_net.requests[0], ie.InferRequest)
    del exec_net
    del ie_core


def test_async_infer_one_req(device):
    ie_core = ie.IECore()
    net = generate_relu_model([1, 3, 32, 32])
    exec_net = ie_core.load_network(net, device, num_requests=1)
    img = generate_image()
    request_handler = exec_net.start_async(request_id=0, inputs={'parameter': img})
    request_handler.wait()
    res = request_handler.output_blobs['relu'].buffer
    assert np.argmax(res) == 531
    del exec_net
    del ie_core


def test_async_infer_many_req(device):
    ie_core = ie.IECore()
    net = generate_relu_model([1, 3, 32, 32])
    exec_net = ie_core.load_network(net, device, num_requests=5)
    img = generate_image()
    for id in range(5):
        request_handler = exec_net.start_async(request_id=id, inputs={'parameter': img})
        request_handler.wait()
        res = request_handler.output_blobs['relu'].buffer
        assert np.argmax(res) == 531
    del exec_net
    del ie_core


def test_async_infer_many_req_get_idle(device):
    ie_core = ie.IECore()
    net = generate_relu_model([1, 3, 32, 32])
    num_requests = 5
    exec_net = ie_core.load_network(net, device, num_requests=num_requests)
    img = generate_image()
    check_id = set()
    for id in range(2*num_requests):
        request_id = exec_net.get_idle_request_id()
        if request_id == -1:
            status = exec_net.wait(num_requests=1, timeout=ie.WaitMode.RESULT_READY)
            assert(status == ie.StatusCode.OK)
        request_id = exec_net.get_idle_request_id()
        assert(request_id >= 0)
        request_handler = exec_net.start_async(request_id=request_id, inputs={'parameter': img})
        check_id.add(request_id)
    status = exec_net.wait(timeout=ie.WaitMode.RESULT_READY)
    assert status == ie.StatusCode.OK
    for id in range(num_requests):
        if id in check_id:
            assert np.argmax(exec_net.requests[id].output_blobs['relu'].buffer) == 531
    del exec_net
    del ie_core


def test_wait_before_start(device):
  ie_core = ie.IECore()
  net = generate_relu_model([1, 3, 32, 32])
  num_requests = 5
  exec_net = ie_core.load_network(net, device, num_requests=num_requests)
  img = generate_image()
  requests = exec_net.requests
  for id in range(num_requests):
      status = requests[id].wait()
      # Plugin API 2.0 has the different behavior will not return this status
      # assert status == ie.StatusCode.INFER_NOT_STARTED
      request_handler = exec_net.start_async(request_id=id, inputs={'parameter': img})
      status = requests[id].wait()
      assert status == ie.StatusCode.OK
      assert np.argmax(request_handler.output_blobs['relu'].buffer) == 531
  del exec_net
  del ie_core


def test_wait_for_callback(device):
    def callback(status, callbacks_info):
        time.sleep(0.01)
        callbacks_info['finished'] += 1

    ie_core = ie.IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    num_requests = 3
    exec_net = ie_core.load_network(net, device, num_requests=num_requests)
    callbacks_info = {}
    callbacks_info['finished'] = 0
    img = generate_image()
    for request in exec_net.requests:
        request.set_completion_callback(callback, callbacks_info)
        request.async_infer({'data': img})

    exec_net.wait(num_requests)
    assert callbacks_info['finished'] == num_requests


def test_wrong_request_id(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie_core.load_network(net, device, num_requests=1)
    img = generate_image()
    with pytest.raises(ValueError) as e:
        exec_net.start_async(request_id=20, inputs={'data': img})
    assert "Incorrect request_id specified!" in str(e.value)
    del exec_net
    del ie_core


def test_wrong_num_requests(device):
    with pytest.raises(ValueError) as e:
        ie_core = ie.IECore()
        net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
        ie_core.load_network(net, device, num_requests=-1)
        assert "Incorrect number of requests specified: -1. Expected positive integer number or zero for auto detection" \
           in str(e.value)
        del ie_core


def test_wrong_num_requests_core(device):
    with pytest.raises(ValueError) as e:
        ie_core = ie.IECore()
        net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
        exec_net = ie_core.load_network(net, device, num_requests=-1)
        assert "Incorrect number of requests specified: -1. Expected positive integer number or zero for auto detection" \
           in str(e.value)
        del ie_core


def test_plugin_accessible_after_deletion(device):
    ie_core = ie.IECore()
    net = generate_relu_model([1, 3, 32, 32])
    exec_net = ie_core.load_network(net, device)
    img = generate_image()
    res = exec_net.infer({'parameter': img})
    assert np.argmax(res['relu'][0]) == 531
    del exec_net
    del ie_core


def test_exec_graph(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie_core.load_network(net, device)
    img = generate_image()
    res = exec_net.infer({'data': img})
    exec_graph = exec_net.get_exec_graph_info()
    exec_graph_file = 'exec_graph.xml'
    exec_graph.serialize(exec_graph_file)
    assert os.path.exists(exec_graph_file)
    os.remove(exec_graph_file)
    del exec_net
    del exec_graph
    del ie_core


def test_export_import(device):
    ie_core = ie.IECore()
    if "EXPORT_IMPORT" not in ie_core.get_metric(device, "OPTIMIZATION_CAPABILITIES"):
        pytest.skip(f"{ie_core.get_metric(device, 'FULL_DEVICE_NAME')} plugin due-to export, import model API isn't implemented.")

    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie_core.load_network(net, "CPU")
    exported_net_file = 'exported_model.bin'
    exec_net.export(exported_net_file)
    assert os.path.exists(exported_net_file)
    exec_net = ie_core.import_network(exported_net_file, "CPU")
    os.remove(exported_net_file)
    img = generate_image()
    res = exec_net.infer({'data': img})
    assert np.argmax(res['fc_out'][0]) == 9
    del exec_net
    del ie_core


def test_multi_out_data(device):
    # Regression test 23965
    # Check that CDataPtr for all output layers not copied  between outputs map items
    ie_core = ie.IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    net.add_outputs(['28/Reshape'])
    exec_net = ie_core.load_network(net, device)
    assert "fc_out" in exec_net.outputs and "28/Reshape" in exec_net.outputs
    assert isinstance(exec_net.outputs["fc_out"], ie.CDataPtr)
    assert isinstance(exec_net.outputs["28/Reshape"], ie.CDataPtr)
    assert exec_net.outputs["fc_out"].name == "fc_out" and exec_net.outputs["fc_out"].shape == [1, 10]
    assert exec_net.outputs["28/Reshape"].name == "28/Reshape" and exec_net.outputs["28/Reshape"].shape == [1, 5184]
    del ie_core
    pass


def test_get_metric(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie_core.load_network(net, device)
    network_name = exec_net.get_metric("NETWORK_NAME")
    assert network_name == "test_model"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU", reason="Device dependent test")
def test_get_config(device):
    ie_core = ie.IECore()
    if ie_core.get_metric(device, "FULL_DEVICE_NAME") == "arm_compute::NEON":
        pytest.skip("Can't run on ARM plugin due-to CPU dependent test")
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie_core.load_network(net, device)
    config = exec_net.get_config("PERF_COUNT")
    assert config == "NO"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "GNA", reason="Device dependent test")
def test_set_config(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie_core.load_network(net, device)
    exec_net.set_config({"DEVICE_MODE" : "GNA_HW"})
    parameter = exec_net.get_config("DEVICE_MODE")
    assert parameter == "GNA_HW"
    exec_net.set_config({"DEVICE_MODE" : "GNA_SW_EXACT"})
    parameter = exec_net.get_config("DEVICE_MODE")
    assert parameter == "GNA_SW_EXACT"


# issue 28996
# checks that objects can deallocate in this order, if not - segfault happends
def test_input_info_deallocation(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie_core.load_network(net, device)
    input_info = exec_net.input_info["data"]
    del ie_core
    del exec_net
    del input_info


def test_outputs_deallocation(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie_core.load_network(net, device)
    output = exec_net.outputs["fc_out"]
    del ie_core
    del exec_net
    del output


def test_exec_graph_info_deallocation(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie_core.load_network(net, device)
    exec_graph_info = exec_net.get_exec_graph_info()
    del ie_core
    del exec_net
    del exec_graph_info
