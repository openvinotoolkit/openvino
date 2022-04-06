# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import os
from sys import platform
from pathlib import Path

import openvino.runtime.opset8 as ov
from openvino.runtime import Model, Core, CompiledModel, Tensor, PartialShape, Extension,\
    tensor_from_file, compile_model

from ..conftest import model_path, model_onnx_path, plugins_path, read_image, \
    get_model_with_template_extension


test_net_xml, test_net_bin = model_path()
test_net_onnx = model_onnx_path()
plugins_xml, plugins_win_xml, plugins_osx_xml = plugins_path()


def test_compact_api_xml():
    img = read_image()

    model = compile_model(test_net_xml)
    assert(isinstance(model, CompiledModel))
    results = model.infer_new_request({"data": img})
    assert np.argmax(results[list(results)[0]]) == 2


def test_compact_api_onnx():
    img = read_image()

    model = compile_model(test_net_onnx)
    assert(isinstance(model, CompiledModel))
    results = model.infer_new_request({"data": img})
    assert np.argmax(results[list(results)[0]]) == 2


def test_core_class():
    input_shape = [1, 3, 4, 4]
    param = ov.parameter(input_shape, np.float32, name="parameter")
    relu = ov.relu(param, name="relu")
    func = Model([relu], [param], "test")
    func.get_ordered_ops()[2].friendly_name = "friendly"

    core = Core()
    model = core.compile_model(func, "CPU", {})

    request = model.create_infer_request()
    input_data = np.random.rand(*input_shape).astype(np.float32) - 0.5

    expected_output = np.maximum(0.0, input_data)

    input_tensor = Tensor(input_data)
    results = request.infer({"parameter": input_tensor})
    assert np.allclose(results[list(results)[0]], expected_output)


def test_compile_model(device):
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(func, device)
    assert isinstance(exec_net, CompiledModel)


def test_compile_model_without_device():
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model)
    assert isinstance(compiled_model, CompiledModel)


def test_read_model_from_ir():
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    assert isinstance(func, Model)

    func = core.read_model(model=test_net_xml)
    assert isinstance(func, Model)


def test_read_model_from_tensor():
    core = Core()
    model = open(test_net_xml).read()
    tensor = tensor_from_file(test_net_bin)
    func = core.read_model(model=model, weights=tensor)
    assert isinstance(func, Model)


def test_read_model_as_path():
    core = Core()
    func = core.read_model(model=Path(test_net_xml), weights=Path(test_net_bin))
    assert isinstance(func, Model)

    func = core.read_model(model=test_net_xml, weights=Path(test_net_bin))
    assert isinstance(func, Model)

    func = core.read_model(model=Path(test_net_xml))
    assert isinstance(func, Model)


def test_read_model_from_onnx():
    core = Core()
    func = core.read_model(model=test_net_onnx)
    assert isinstance(func, Model)


def test_read_model_from_onnx_as_path():
    core = Core()
    func = core.read_model(model=Path(test_net_onnx))
    assert isinstance(func, Model)


def test_read_net_from_buffer():
    core = Core()
    with open(test_net_bin, "rb") as f:
        bin = f.read()
    with open(model_path()[0], "rb") as f:
        xml = f.read()
    func = core.read_model(model=xml, weights=bin)
    assert isinstance(func, Model)


def test_net_from_buffer_valid():
    core = Core()
    with open(test_net_bin, "rb") as f:
        bin = f.read()
    with open(model_path()[0], "rb") as f:
        xml = f.read()
    func = core.read_model(model=xml, weights=bin)
    ref_func = core.read_model(model=test_net_xml, weights=test_net_bin)
    assert func.get_parameters() == ref_func.get_parameters()
    assert func.get_results() == ref_func.get_results()
    assert func.get_ordered_ops() == ref_func.get_ordered_ops()


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


def test_get_property():
    ie = Core()
    conf = ie.get_property("CPU", "CPU_BIND_THREAD")
    assert conf == "YES"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU",
                    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test")
def test_get_property_list_of_str():
    ie = Core()
    param = ie.get_property("CPU", "OPTIMIZATION_CAPABILITIES")
    assert isinstance(param, list), "Parameter value for 'OPTIMIZATION_CAPABILITIES' " \
                                    f"metric must be a list but {type(param)} is returned"
    assert all(isinstance(v, str) for v in param), \
        "Not all of the parameter values for 'OPTIMIZATION_CAPABILITIES' metric are strings!"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU",
                    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test")
def test_get_property_tuple_of_two_ints():
    ie = Core()
    param = ie.get_property("CPU", "RANGE_FOR_STREAMS")
    assert isinstance(param, tuple), "Parameter value for 'RANGE_FOR_STREAMS' " \
                                     f"metric must be tuple but {type(param)} is returned"
    assert all(isinstance(v, int) for v in param), \
        "Not all of the parameter values for 'RANGE_FOR_STREAMS' metric are integers!"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU",
                    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test")
def test_get_property_tuple_of_three_ints():
    ie = Core()
    param = ie.get_property("CPU", "RANGE_FOR_ASYNC_INFER_REQUESTS")
    assert isinstance(param, tuple), "Parameter value for 'RANGE_FOR_ASYNC_INFER_REQUESTS' " \
                                     f"metric must be tuple but {type(param)} is returned"
    assert all(isinstance(v, int) for v in param), "Not all of the parameter values for " \
                                                   "'RANGE_FOR_ASYNC_INFER_REQUESTS' metric are integers!"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU",
                    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test")
def test_get_property_str():
    ie = Core()
    param = ie.get_property("CPU", "FULL_DEVICE_NAME")
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
    ie.register_plugin("openvino_intel_cpu_plugin", "BLA")
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(func, "BLA")
    assert isinstance(exec_net, CompiledModel), \
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
                      CompiledModel), "Cannot load the network to " \
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


@pytest.mark.template_extension
def test_add_extension_template_extension(device):
    core, model = get_model_with_template_extension()
    assert isinstance(model, Model)

    before_reshape = PartialShape([1, 3, 22, 22])
    after_reshape = PartialShape([8, 9, 33, 66])
    new_shapes = {"in_data": after_reshape}
    assert model.input().partial_shape == before_reshape
    model.reshape(new_shapes)
    # compile to check objects can be destroyed
    # in order core -> model -> compiled
    compiled = core.compile_model(model, device)
    assert compiled.input().partial_shape == after_reshape


def test_add_extension():
    class EmptyExtension(Extension):
        def __init__(self) -> None:
            super().__init__()

    core = Core()
    core.add_extension(EmptyExtension())
    core.add_extension([EmptyExtension(), EmptyExtension()])
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    assert isinstance(model, Model)


def test_read_model_from_buffer_no_weights(device):
    model = bytes(b"""<net name="add_model" version="10">
    <layers>
    <layer id="0" name="x" type="Parameter" version="opset1">
        <data element_type="f32" shape="3,4,5"/>
        <output>
            <port id="0" precision="FP32">
                <dim>3</dim>
                <dim>4</dim>
                <dim>5</dim>
            </port>
        </output>
    </layer>
    <layer id="1" name="y" type="Parameter" version="opset1">
        <data element_type="f32" shape="3,4,5"/>
        <output>
            <port id="0" precision="FP32">
                <dim>3</dim>
                <dim>4</dim>
                <dim>5</dim>
            </port>
        </output>
    </layer>
    <layer id="2" name="sum" type="Add" version="opset1">
        <input>
            <port id="0">
                <dim>3</dim>
                <dim>4</dim>
                <dim>5</dim>
            </port>
            <port id="1">
                <dim>3</dim>
                <dim>4</dim>
                <dim>5</dim>
            </port>
        </input>
        <output>
            <port id="2" precision="FP32">
                <dim>3</dim>
                <dim>4</dim>
                <dim>5</dim>
            </port>
        </output>
    </layer>
    <layer id="3" name="sum/sink_port_0" type="Result" version="opset1">
        <input>
            <port id="0">
                <dim>3</dim>
                <dim>4</dim>
                <dim>5</dim>
            </port>
        </input>
    </layer>
    </layers>
    <edges>
    <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
    <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
    <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
    </edges>
</net>""")
    core = Core()
    func = core.read_model(model=model)
    assert isinstance(func, Model)


def test_infer_new_request_return_type(device):
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    img = read_image()
    exec_net = ie.compile_model(func, device)
    res = exec_net.infer_new_request({"data": img})
    arr = res[list(res)[0]][0]

    assert isinstance(arr, np.ndarray)
    assert arr.itemsize == 4
    assert arr.shape == (10,)
    assert arr.dtype == "float32"
    assert arr.nbytes == 40
