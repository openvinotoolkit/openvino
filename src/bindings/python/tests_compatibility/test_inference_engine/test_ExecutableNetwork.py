# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os
import pytest
import time

from openvino.inference_engine import ie_api as ie
from tests_compatibility.conftest import model_path
from tests_compatibility.test_utils.test_utils import generate_image


is_myriad = os.environ.get("TEST_DEVICE") == "MYRIAD"
test_net_xml, test_net_bin = model_path(is_myriad)


def test_infer(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie_core.load_network(net, device)
    img = generate_image()
    res = exec_net.infer({'data': img})
    assert np.argmax(res['fc_out'][0]) == 9
    del exec_net
    del ie_core


def test_infer_net_from_buffer(device):
    ie_core = ie.IECore()
    with open(test_net_bin, 'rb') as f:
        bin = f.read()
    with open(test_net_xml, 'rb') as f:
        xml = f.read()
    net = ie_core.read_network(model=xml, weights=bin, init_from_buffer=True)
    net2 = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie_core.load_network(net, device)
    exec_net2 = ie_core.load_network(net2, device)
    img = generate_image()
    res = exec_net.infer({'data': img})
    res2 = exec_net2.infer({'data': img})
    del ie_core
    del exec_net
    del exec_net2
    assert np.allclose(res['fc_out'], res2['fc_out'], atol=1E-4, rtol=1E-4)


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
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie_core.load_network(net, device, num_requests=1)
    img = generate_image()
    request_handler = exec_net.start_async(request_id=0, inputs={'data': img})
    request_handler.wait()
    res = request_handler.output_blobs['fc_out'].buffer
    assert np.argmax(res) == 9
    del exec_net
    del ie_core


def test_async_infer_many_req(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie_core.load_network(net, device, num_requests=5)
    img = generate_image()
    for id in range(5):
        request_handler = exec_net.start_async(request_id=id, inputs={'data': img})
        request_handler.wait()
        res = request_handler.output_blobs['fc_out'].buffer
        assert np.argmax(res) == 9
    del exec_net
    del ie_core


def test_async_infer_many_req_get_idle(device):
    ie_core = ie.IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
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
        request_handler = exec_net.start_async(request_id=request_id, inputs={'data': img})
        check_id.add(request_id)
    status = exec_net.wait(timeout=ie.WaitMode.RESULT_READY)
    assert status == ie.StatusCode.OK
    for id in range(num_requests):
        if id in check_id:
            assert np.argmax(exec_net.requests[id].output_blobs['fc_out'].buffer) == 9
    del exec_net
    del ie_core


def test_wait_before_start(device):
  ie_core = ie.IECore()
  net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
  num_requests = 5
  exec_net = ie_core.load_network(net, device, num_requests=num_requests)
  img = generate_image()
  requests = exec_net.requests
  for id in range(num_requests):
      status = requests[id].wait()
      assert status == ie.StatusCode.INFER_NOT_STARTED
      request_handler = exec_net.start_async(request_id=id, inputs={'data': img})
      status = requests[id].wait()
      assert status == ie.StatusCode.OK
      assert np.argmax(request_handler.output_blobs['fc_out'].buffer) == 9
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
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie_core.load_network(net, device)
    img = generate_image()
    res = exec_net.infer({'data': img})
    assert np.argmax(res['fc_out'][0]) == 9
    del exec_net
    del ie_core


def test_exec_graph(device):
    ie_core = ie.IECore()
    if device == "CPU":
        if ie_core.get_metric(device, "FULL_DEVICE_NAME") == "arm_compute::NEON":
            pytest.skip("Can't run on ARM plugin due-to get_exec_graph_info method isn't implemented")
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


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "MYRIAD",
                    reason="Device specific test. Only MYRIAD plugin implements network export")
def test_export_import():
    ie_core = ie.IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie_core.load_network(net, "MYRIAD")
    exported_net_file = 'exported_model.bin'
    exec_net.export(exported_net_file)
    assert os.path.exists(exported_net_file)
    exec_net = ie_core.import_network(exported_net_file, "MYRIAD")
    os.remove(exported_net_file)
    img = generate_image()
    res = exec_net.infer({'data': img})
    assert np.argmax(res['fc_out'][0]) == 3
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
    if device == "CPU":
        if ie_core.get_metric(device, "FULL_DEVICE_NAME") == "arm_compute::NEON":
            pytest.skip("Can't run on ARM plugin due-to get_exec_graph_info method isn't implemented")
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie_core.load_network(net, device)
    exec_graph_info = exec_net.get_exec_graph_info()
    del ie_core
    del exec_net
    del exec_graph_info
