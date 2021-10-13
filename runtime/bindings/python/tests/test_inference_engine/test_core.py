# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import os
from sys import platform
from pathlib import Path

import ngraph as ng
import openvino as ov
from ngraph.impl import Function, Shape, Type
from ngraph.impl.op import Parameter
from openvino import TensorDesc, Blob

from ..conftest import model_path, model_onnx_path, plugins_path

test_net_xml, test_net_bin = model_path()
test_net_onnx = model_onnx_path()
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


@pytest.mark.skip(reason="Fix")
def test_ie_core_class():
    input_shape = [1, 3, 4, 4]
    param = ng.parameter(input_shape, np.float32, name="parameter")
    relu = ng.relu(param, name="relu")
    func = Function([relu], [param], "test")
    func.get_ordered_ops()[2].friendly_name = "friendly"

    cnn_network = ov.IENetwork(func)

    ie_core = ov.Core()
    ie_core.set_config({}, device_name="CPU")
    executable_network = ie_core.compile_model(cnn_network, "CPU", {})

    td = TensorDesc("FP32", input_shape, "NCHW")

    # from IPython import embed; embed()

    request = executable_network.create_infer_request()
    input_data = np.random.rand(*input_shape) - 0.5

    expected_output = np.maximum(0.0, input_data)

    input_blob = Blob(td, input_data)

    request.set_input({"parameter": input_blob})
    request.infer()

    result = request.get_blob("relu").buffer

    assert np.allclose(result, expected_output)


def test_compile_model(device):
    ie = ov.Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(func, device)
    assert isinstance(exec_net, ov.ExecutableNetwork)


def test_read_model():
    ie_core = ov.Core()
    func = ie_core.read_model(model=test_net_xml, weights=test_net_bin)
    assert isinstance(func, ov.IENetwork)

    func = ie_core.read_model(model=test_net_xml)
    assert isinstance(func, ov.IENetwork)


def test_read_model_from_blob():
    ie_core = ov.Core()
    model = open(test_net_xml).read()
    blob = ov.blob_from_file(test_net_bin)
    func = ie_core.read_model(model=model, blob=blob)
    assert isinstance(func, ov.IENetwork)


def test_read_model_from_blob_valid():
    ie_core = ov.Core()
    model = open(test_net_xml).read()
    blob = ov.blob_from_file(test_net_bin)
    func = ie_core.read_model(model=model, blob=blob)
    ref_func = ie_core.read_model(model=test_net_xml, weights=test_net_bin)
    assert func.name == ref_func.name
    assert func.batch_size == ref_func.batch_size
    ii_func = func.input_info
    ii_func2 = ref_func.input_info
    o_func = func.outputs
    o_func2 = ref_func.outputs
    assert ii_func.keys() == ii_func2.keys()
    assert o_func.keys() == o_func2.keys()


def test_read_model_as_path():
    ie_core = ov.Core()
    func = ie_core.read_model(model=Path(test_net_xml), weights=Path(test_net_bin))
    assert isinstance(func, ov.IENetwork)

    func = ie_core.read_model(model=test_net_xml, weights=Path(test_net_bin))
    assert isinstance(func, ov.IENetwork)

    func = ie_core.read_model(model=Path(test_net_xml))
    assert isinstance(func, ov.IENetwork)


def test_read_model_from_onnx():
    ie_core = ov.Core()
    func = ie_core.read_model(model=test_net_onnx)
    assert isinstance(func, ov.IENetwork)


def test_read_model_from_onnx_as_path():
    ie_core = ov.Core()
    func = ie_core.read_model(model=Path(test_net_onnx))
    assert isinstance(func, ov.IENetwork)


def test_read_net_from_buffer():
    ie_core = ov.Core()
    with open(test_net_bin, "rb") as f:
        bin = f.read()
    with open(model_path()[0], "rb") as f:
        xml = f.read()
    func = ie_core.read_model(model=xml, weights=bin)
    assert isinstance(func, ov.IENetwork)


def test_net_from_buffer_valid():
    ie_core = ov.Core()
    with open(test_net_bin, "rb") as f:
        bin = f.read()
    with open(model_path()[0], "rb") as f:
        xml = f.read()
    func = ie_core.read_model(model=xml, weights=bin)
    ref_func = ie_core.read_model(model=test_net_xml, weights=test_net_bin)
    assert func.name == ref_func.name
    assert func.batch_size == ref_func.batch_size
    ii_func = func.input_info
    ii_func2 = ref_func.input_info
    o_func = func.outputs
    o_func2 = ref_func.outputs
    assert ii_func.keys() == ii_func2.keys()
    assert o_func.keys() == o_func2.keys()


def test_get_version(device):
    ie = ov.Core()
    version = ie.get_versions(device)
    assert isinstance(version, dict), "Returned version must be a dictionary"
    assert device in version, "{} plugin version wasn't found in versions"
    assert hasattr(version[device], "major"), "Returned version has no field 'major'"
    assert hasattr(version[device], "minor"), "Returned version has no field 'minor'"
    assert hasattr(version[device], "description"), "Returned version has no field 'description'"
    assert hasattr(version[device], "build_number"), "Returned version has no field 'build_number'"


def test_available_devices(device):
    ie = ov.Core()
    devices = ie.available_devices
    assert device in devices, f"Current device '{device}' is not listed in " \
                              f"available devices '{', '.join(devices)}'"


def test_get_config():
    ie = ov.Core()
    conf = ie.get_config("CPU", "CPU_BIND_THREAD")
    assert conf == "YES"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU",
                    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test")
def test_get_metric_list_of_str():
    ie = ov.Core()
    param = ie.get_metric("CPU", "OPTIMIZATION_CAPABILITIES")
    assert isinstance(param, list), "Parameter value for 'OPTIMIZATION_CAPABILITIES' " \
                                    f"metric must be a list but {type(param)} is returned"
    assert all(isinstance(v, str) for v in param), \
        "Not all of the parameter values for 'OPTIMIZATION_CAPABILITIES' metric are strings!"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU",
                    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test")
def test_get_metric_tuple_of_two_ints():
    ie = ov.Core()
    param = ie.get_metric("CPU", "RANGE_FOR_STREAMS")
    assert isinstance(param, tuple), "Parameter value for 'RANGE_FOR_STREAMS' " \
                                     f"metric must be tuple but {type(param)} is returned"
    assert all(isinstance(v, int) for v in param), \
        "Not all of the parameter values for 'RANGE_FOR_STREAMS' metric are integers!"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU",
                    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test")
def test_get_metric_tuple_of_three_ints():
    ie = ov.Core()
    param = ie.get_metric("CPU", "RANGE_FOR_ASYNC_INFER_REQUESTS")
    assert isinstance(param, tuple), "Parameter value for 'RANGE_FOR_ASYNC_INFER_REQUESTS' " \
                                     f"metric must be tuple but {type(param)} is returned"
    assert all(isinstance(v, int) for v in param), "Not all of the parameter values for " \
                                                   "'RANGE_FOR_ASYNC_INFER_REQUESTS' metric are integers!"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU",
                    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test")
def test_get_metric_str():
    ie = ov.Core()
    param = ie.get_metric("CPU", "FULL_DEVICE_NAME")
    assert isinstance(param, str), "Parameter value for 'FULL_DEVICE_NAME' " \
                                   f"metric must be string but {type(param)} is returned"


def test_query_model(device):
    ie = ov.Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    query_res = ie.query_model(model=func, device_name=device)
    ops_net = func.get_ordered_ops()
    ops_net_names = [op.friendly_name for op in ops_net]
    assert [key for key in query_res.keys() if key not in ops_net_names] == [], \
        "Not all network layers present in query_model results"
    assert next(iter(set(query_res.values()))) == device, "Wrong device for some layers"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU", reason="Device independent test")
def test_register_plugin():
    ie = ov.Core()
    ie.register_plugin("MKLDNNPlugin", "BLA")
    net = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(net, "BLA")
    assert isinstance(exec_net, ov.ExecutableNetwork), \
        "Cannot load the network to the registered plugin with name 'BLA'"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU", reason="Device independent test")
def test_register_plugins():
    ie = ov.Core()
    if platform == "linux" or platform == "linux2":
        ie.register_plugins(plugins_xml)
    elif platform == "darwin":
        ie.register_plugins(plugins_osx_xml)
    elif platform == "win32":
        ie.register_plugins(plugins_win_xml)

    net = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(net, "CUSTOM")
    assert isinstance(exec_net,
                      ov.ExecutableNetwork), "Cannot load the network to " \
                                             "the registered plugin with name 'CUSTOM' " \
                                             "registred in the XML file"


def test_create_IENetwork_from_nGraph():
    element_type = Type.f32
    param = Parameter(element_type, Shape([1, 3, 22, 22]))
    relu = ng.relu(param)
    func = Function([relu], [param], "test")
    cnnNetwork = ov.IENetwork(func)
    assert cnnNetwork is not None
    func2 = cnnNetwork.get_function()
    assert func2 is not None
    assert len(func2.get_ops()) == 3
