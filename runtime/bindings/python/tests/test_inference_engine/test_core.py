# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import os
from sys import platform
from pathlib import Path

import openvino.opset8 as ov
from openvino import Core, IENetwork, ExecutableNetwork, tensor_from_file
from openvino.impl import Function
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
def test_core_class():
    input_shape = [1, 3, 4, 4]
    param = ov.parameter(input_shape, np.float32, name="parameter")
    relu = ov.relu(param, name="relu")
    func = Function([relu], [param], "test")
    func.get_ordered_ops()[2].friendly_name = "friendly"

    cnn_network = IENetwork(func)

    core = Core()
    core.set_config({}, device_name="CPU")
    executable_network = core.compile_model(cnn_network, "CPU", {})

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
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(func, device)
    assert isinstance(exec_net, ExecutableNetwork)


def test_read_model_from_ir():
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    assert isinstance(func, Function)

    func = core.read_model(model=test_net_xml)
    assert isinstance(func, Function)


def test_read_model_from_tensor():
    core = Core()
    model = open(test_net_xml).read()
    tensor = tensor_from_file(test_net_bin)
    func = core.read_model(model=model, weights=tensor)
    assert isinstance(func, Function)


def test_read_model_as_path():
    core = Core()
    func = core.read_model(model=Path(test_net_xml), weights=Path(test_net_bin))
    assert isinstance(func, Function)

    func = core.read_model(model=test_net_xml, weights=Path(test_net_bin))
    assert isinstance(func, Function)

    func = core.read_model(model=Path(test_net_xml))
    assert isinstance(func, Function)


def test_read_model_from_onnx():
    core = Core()
    func = core.read_model(model=test_net_onnx)
    assert isinstance(func, Function)


def test_read_model_from_onnx_as_path():
    core = Core()
    func = core.read_model(model=Path(test_net_onnx))
    assert isinstance(func, Function)


@pytest.mark.xfail("68212")
def test_read_net_from_buffer():
    core = Core()
    with open(test_net_bin, "rb") as f:
        bin = f.read()
    with open(model_path()[0], "rb") as f:
        xml = f.read()
    func = core.read_model(model=xml, weights=bin)
    assert isinstance(func, IENetwork)


@pytest.mark.xfail("68212")
def test_net_from_buffer_valid():
    core = Core()
    with open(test_net_bin, "rb") as f:
        bin = f.read()
    with open(model_path()[0], "rb") as f:
        xml = f.read()
    func = core.read_model(model=xml, weights=bin)
    ref_func = core.read_model(model=test_net_xml, weights=test_net_bin)
    assert func.name == func.name
    assert func.batch_size == ref_func.batch_size
    ii_func = func.input_info
    ii_func2 = ref_func.input_info
    o_func = func.outputs
    o_func2 = ref_func.outputs
    assert ii_func.keys() == ii_func2.keys()
    assert o_func.keys() == o_func2.keys()


def test_get_version(device):
    ie = Core()
    version = ie.get_versions(device)
    assert isinstance(version, dict), "Returned version must be a dictionary"
    assert device in version, "{} plugin version wasn't found in versions"
    assert hasattr(version[device], "major"), "Returned version has no field 'major'"
    assert hasattr(version[device], "minor"), "Returned version has no field 'minor'"
    assert hasattr(version[device], "description"), "Returned version has no field 'description'"
    assert hasattr(version[device], "build_number"), "Returned version has no field 'build_number'"


def test_available_devices(device):
    ie = Core()
    devices = ie.available_devices
    assert device in devices, f"Current device '{device}' is not listed in " \
                              f"available devices '{', '.join(devices)}'"


def test_get_config():
    ie = Core()
    conf = ie.get_config("CPU", "CPU_BIND_THREAD")
    assert conf == "YES"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU",
                    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test")
def test_get_metric_list_of_str():
    ie = Core()
    param = ie.get_metric("CPU", "OPTIMIZATION_CAPABILITIES")
    assert isinstance(param, list), "Parameter value for 'OPTIMIZATION_CAPABILITIES' " \
                                    f"metric must be a list but {type(param)} is returned"
    assert all(isinstance(v, str) for v in param), \
        "Not all of the parameter values for 'OPTIMIZATION_CAPABILITIES' metric are strings!"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU",
                    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test")
def test_get_metric_tuple_of_two_ints():
    ie = Core()
    param = ie.get_metric("CPU", "RANGE_FOR_STREAMS")
    assert isinstance(param, tuple), "Parameter value for 'RANGE_FOR_STREAMS' " \
                                     f"metric must be tuple but {type(param)} is returned"
    assert all(isinstance(v, int) for v in param), \
        "Not all of the parameter values for 'RANGE_FOR_STREAMS' metric are integers!"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU",
                    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test")
def test_get_metric_tuple_of_three_ints():
    ie = Core()
    param = ie.get_metric("CPU", "RANGE_FOR_ASYNC_INFER_REQUESTS")
    assert isinstance(param, tuple), "Parameter value for 'RANGE_FOR_ASYNC_INFER_REQUESTS' " \
                                     f"metric must be tuple but {type(param)} is returned"
    assert all(isinstance(v, int) for v in param), "Not all of the parameter values for " \
                                                   "'RANGE_FOR_ASYNC_INFER_REQUESTS' metric are integers!"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU",
                    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test")
def test_get_metric_str():
    ie = Core()
    param = ie.get_metric("CPU", "FULL_DEVICE_NAME")
    assert isinstance(param, str), "Parameter value for 'FULL_DEVICE_NAME' " \
                                   f"metric must be string but {type(param)} is returned"


def test_query_model(device):
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    query_res = ie.query_model(model=func, device_name=device)
    ops_func = func.get_ordered_ops()
    ops_func_names = [op.friendly_name for op in ops_func]
    assert [key for key in query_res.keys() if key not in ops_func_names] == [], \
        "Not all network layers present in query_model results"
    assert next(iter(set(query_res.values()))) == device, "Wrong device for some layers"


@pytest.mark.dynamic_library
@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU", reason="Device independent test")
def test_register_plugin():
    ie = Core()
    ie.register_plugin("MKLDNNPlugin", "BLA")
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(func, "BLA")
    assert isinstance(exec_net, ExecutableNetwork), \
        "Cannot load the network to the registered plugin with name 'BLA'"


@pytest.mark.dynamic_library
@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU", reason="Device independent test")
def test_register_plugins():
    ie = Core()
    if platform == "linux" or platform == "linux2":
        ie.register_plugins(plugins_xml)
    elif platform == "darwin":
        ie.register_plugins(plugins_osx_xml)
    elif platform == "win32":
        ie.register_plugins(plugins_win_xml)

    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(func, "CUSTOM")
    assert isinstance(exec_net,
                      ExecutableNetwork), "Cannot load the network to " \
                                          "the registered plugin with name 'CUSTOM' " \
                                          "registered in the XML file"


@pytest.mark.skip(reason="Need to figure out if it's expected behaviour (fails with C++ API as well")
def test_unregister_plugin(device):
    ie = Core()
    ie.unload_plugin(device)
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    with pytest.raises(RuntimeError) as e:
        ie.load_network(func, device)
    assert f"Device with '{device}' name is not registered in the InferenceEngine" in str(e.value)


@pytest.mark.xfail("68212")
@pytest.mark.template_extension
def test_add_extension(device):
    model = bytes(b"""<net name="Network" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="2,2,2,1"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer name="operation" id="1" type="Template" version="custom_opset">
            <data  add="11"/>
            <input>
                <port id="1" precision="FP32">
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>""")

    core = Core()
    if platform == "win32":
        core.add_extension(extension_path="template_extension.dll")
    else:
        core.add_extension(extension_path="libtemplate_extension.so")
    func = core.read_model(model=model, init_from_buffer=True)
    assert isinstance(func, Function)

    # input_blob = next(iter(network.input_info))
    # n, c, h, w = network.input_info[input_blob].input_data.shape

    # input_values = np.ndarray(buffer=np.array([1, 2, 3, 4, 5, 6, 7, 8]), shape = (n, c, h, w), dtype=int)
    # expected = np.ndarray(buffer=np.array([12, 13, 14, 15, 16, 17, 18, 19]),
    # shape = (n, c, h, w), dtype=int)
    #
    # exec_network = core.compile_model(func, device)
    # computed = exec_network.infer_new_request(inputs={input_blob : input_values})
    # output_blob = next(iter(network.outputs))
    # assert np.allclose(expected, computed[output_blob], atol=1e-2, rtol=1e-2)
