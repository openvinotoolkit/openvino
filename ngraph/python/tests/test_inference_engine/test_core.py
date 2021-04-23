# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import ngraph as ng
from ngraph.impl.op import Parameter
from ngraph.impl import Function, Shape, Type
from openvino.inference_engine import IECore, TensorDesc, Blob, IENetwork, ExecutableNetwork
from openvino.inference_engine.ie_api import blob_from_file
from ..conftest import model_path, plugins_path
import os
import pytest
from sys import platform


test_net_xml, test_net_bin = model_path()
plugins_xml, plugins_win_xml, plugins_osx_xml = plugins_path()

def test_blobs():
    input_shape = [1, 3, 4, 4]
    input_data_float32 = (np.random.rand(*input_shape) - 0.5).astype(np.float32)

    td = TensorDesc("FP32", input_shape, "NCHW")

    input_blob_float32 = Blob(td, input_data_float32)

    assert np.all(np.equal(input_blob_float32.buffer, input_data_float32))

    input_data_int16 = (np.random.rand(*input_shape) + 0.5).astype(np.int16)

    td = TensorDesc("I16", input_shape, "NCHW")

    input_blob_i16 = Blob(td, input_data_int16)

    assert np.all(np.equal(input_blob_i16.buffer, input_data_int16))


def test_ie_core_class():
    input_shape = [1, 3, 4, 4]
    param = ng.parameter(input_shape, np.float32, name="parameter")
    relu = ng.relu(param, name="relu")
    func = Function([relu], [param], 'test')
    func.get_ordered_ops()[2].friendly_name = "friendly"

    capsule = Function.to_capsule(func)
    cnn_network = IENetwork(capsule)

    ie_core = IECore()
    ie_core.set_config({}, device_name='CPU')
    executable_network = ie_core.load_network(cnn_network, 'CPU', {})

    td = TensorDesc("FP32", input_shape, "NCHW")

    # from IPython import embed; embed()

    request = executable_network.create_infer_request()
    input_data = np.random.rand(*input_shape) - 0.5

    expected_output = np.maximum(0.0, input_data)

    input_blob = Blob(td, input_data)

    request.set_input({'parameter': input_blob})
    request.infer()

    result = request.get_blob('relu').buffer

    assert np.allclose(result, expected_output)


def test_load_network(device):
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.load_network(net, device)
    assert isinstance(exec_net, ExecutableNetwork)


def test_read_network():
    ie_core = IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    assert isinstance(net, IENetwork)


def test_read_network_from_memory():
    ie_core = IECore()
    model = open(test_net_xml).read()
    blob = blob_from_file(test_net_bin)
    net = ie_core.read_network(model, blob)
    assert isinstance(net, IENetwork) 


def test_get_version(device):
    ie = IECore()
    version = ie.get_versions(device)
    assert isinstance(version, dict), "Returned version must be a dictionary"
    assert device in version, "{} plugin version wasn't found in versions"
    assert hasattr(version[device], "major"), "Returned version has no field 'major'"
    assert hasattr(version[device], "minor"), "Returned version has no field 'minor'"
    assert hasattr(version[device], "description"), "Returned version has no field 'description'"
    assert hasattr(version[device], "build_number"), "Returned version has no field 'build_number'"


def test_available_devices(device):
    ie = IECore()
    devices = ie.available_devices
    assert device in devices, "Current device '{}' is not listed in available devices '{}'".format(device,
                                                                                                   ', '.join(devices))

def test_get_config():
    ie = IECore()
    conf = ie.get_config("CPU", "CPU_BIND_THREAD")
    assert conf == "YES"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU",
                    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test")
def test_get_metric_list_of_str():
    ie = IECore()
    param = ie.get_metric("CPU", "OPTIMIZATION_CAPABILITIES")
    assert isinstance(param, list), "Parameter value for 'OPTIMIZATION_CAPABILITIES' " \
                                    f"metric must be a list but {type(param)} is returned"
    assert all(isinstance(v, str) for v in param), "Not all of the parameter values for 'OPTIMIZATION_CAPABILITIES' " \
                                                   "metric are strings!"



@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU",
                    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test")
def test_get_metric_tuple_of_two_ints():
    ie = IECore()
    param = ie.get_metric("CPU", "RANGE_FOR_STREAMS")
    assert isinstance(param, tuple), "Parameter value for 'RANGE_FOR_STREAMS' " \
                                     f"metric must be tuple but {type(param)} is returned"
    assert all(isinstance(v, int) for v in param), "Not all of the parameter values for 'RANGE_FOR_STREAMS' " \
                                                   "metric are integers!"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU",
                    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test")
def test_get_metric_tuple_of_three_ints():
    ie = IECore()
    param = ie.get_metric("CPU", "RANGE_FOR_ASYNC_INFER_REQUESTS")
    assert isinstance(param, tuple), "Parameter value for 'RANGE_FOR_ASYNC_INFER_REQUESTS' " \
                                     f"metric must be tuple but {type(param)} is returned"
    assert all(isinstance(v, int) for v in param), "Not all of the parameter values for " \
                                                   "'RANGE_FOR_ASYNC_INFER_REQUESTS' metric are integers!"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU",
                    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test")
def test_get_metric_str():
    ie = IECore()
    param = ie.get_metric("CPU", "FULL_DEVICE_NAME")
    assert isinstance(param, str), "Parameter value for 'FULL_DEVICE_NAME' " \
                                   f"metric must be string but {type(param)} is returned"


def test_query_network(device):
    import ngraph as ng
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    query_res = ie.query_network(network=net, device_name=device)
    func_net = ng.function_from_cnn(net)
    ops_net = func_net.get_ordered_ops()
    ops_net_names = [op.friendly_name for op in ops_net]
    assert [key for key in query_res.keys() if key not in ops_net_names] == [], \
        "Not all network layers present in query_network results"
    assert next(iter(set(query_res.values()))) == device, "Wrong device for some layers"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU", reason="Device independent test")
def test_register_plugin():
    ie = IECore()
    ie.register_plugin("MKLDNNPlugin", "BLA")
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.load_network(net, "BLA")
    assert isinstance(exec_net, ExecutableNetwork), "Cannot load the network to the registered plugin with name 'BLA'"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU", reason="Device independent test")
def test_register_plugins():
    ie = IECore()
    if platform == "linux" or platform == "linux2":
        ie.register_plugins(plugins_xml)
    elif platform == "darwin":
        ie.register_plugins(plugins_osx_xml)
    elif platform == "win32":
        ie.register_plugins(plugins_win_xml)

    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.load_network(net, "CUSTOM")
    assert isinstance(exec_net,
                      ExecutableNetwork), "Cannot load the network to the registered plugin with name 'CUSTOM' " \
                                          "registred in the XML file"


def test_create_IENetwork_from_nGraph():
    element_type = Type.f32
    param = Parameter(element_type, Shape([1, 3, 22, 22]))
    relu = ng.relu(param)
    func = Function([relu], [param], 'test')
    cnnNetwork = IENetwork(func)
    assert cnnNetwork != None
    func2 = cnnNetwork.get_function()
    assert func2 != None
    assert len(func2.get_ops()) == 3
