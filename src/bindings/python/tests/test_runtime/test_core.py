# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import sys
import numpy as np
import os
from pathlib import Path

from openvino import (
    Model,
    Core,
    Tensor,
    PartialShape,
    CompiledModel,
    tensor_from_file,
    compile_model,
)

from openvino.runtime import Extension

from tests.conftest import (
    model_path,
    model_onnx_path,
    get_model_with_template_extension,
)

from tests.test_utils.test_utils import (
    generate_image,
    generate_relu_compiled_model,
    get_relu_model,
    plugins_path,
    compare_models,
)


test_net_xml, test_net_bin = model_path()
test_net_onnx = model_onnx_path()


def test_compact_api_xml():
    img = generate_image()

    compiled_model = compile_model(get_relu_model())
    assert isinstance(compiled_model, CompiledModel)
    results = compiled_model.infer_new_request({"data": img})
    assert np.argmax(results[list(results)[0]]) == 531


def test_compact_api_xml_posix_path():
    compiled_model = compile_model(Path(test_net_xml))
    assert isinstance(compiled_model, CompiledModel)


def test_compact_api_wrong_path():
    # as inner method takes py::object as an input and turns it into string
    # it is necessary to assure that provided argument is either
    # python string or pathlib.Path object rather than some class
    # with implemented __str__ magic method
    class TestClass:
        def __str__(self):
            return "test class"
    with pytest.raises(RuntimeError) as e:
        compile_model(TestClass())
    assert "Path: 'test class' does not exist. Please provide valid model's path either as a string, bytes or pathlib.Path" in str(e.value)


def test_core_class(device):
    input_shape = [1, 3, 4, 4]
    compiled_model = generate_relu_compiled_model(device, input_shape=input_shape)

    request = compiled_model.create_infer_request()
    input_data = np.random.rand(*input_shape).astype(np.float32) - 0.5

    expected_output = np.maximum(0.0, input_data)

    input_tensor = Tensor(input_data)
    results = request.infer({"data": input_tensor})
    assert np.allclose(results[list(results)[0]], expected_output)


def test_compile_model(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, device)
    assert isinstance(compiled_model, CompiledModel)


def test_compile_model_without_device():
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model)
    assert isinstance(compiled_model, CompiledModel)


def test_read_model_from_ir():
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    assert isinstance(model, Model)

    model = core.read_model(model=test_net_xml)
    assert isinstance(model, Model)


def test_read_model_from_tensor():
    core = Core()
    model = open(test_net_xml).read()
    tensor = tensor_from_file(test_net_bin)
    model = core.read_model(model=model, weights=tensor)
    assert isinstance(model, Model)


def test_read_model_with_wrong_input():
    core = Core()
    with pytest.raises(RuntimeError) as e:
        core.read_model(model=3, weights=3)
    assert "Provided python object type <class 'int'> isn't supported as 'model' argument." in str(e.value)


def test_read_model_as_path():
    core = Core()
    model = core.read_model(model=Path(test_net_xml), weights=Path(test_net_bin))
    assert isinstance(model, Model)

    model = core.read_model(model=test_net_xml, weights=Path(test_net_bin))
    assert isinstance(model, Model)

    model = core.read_model(model=Path(test_net_xml))
    assert isinstance(model, Model)


def test_read_model_from_onnx():
    core = Core()
    model = core.read_model(model=test_net_onnx)
    assert isinstance(model, Model)


def test_read_model_from_onnx_as_path():
    core = Core()
    model = core.read_model(model=Path(test_net_onnx))
    assert isinstance(model, Model)


def test_read_model_from_buffer():
    core = Core()
    with open(test_net_bin, "rb") as f:
        weights = f.read()
    with open(model_path()[0], "rb") as f:
        xml = f.read()
    model = core.read_model(model=xml, weights=weights)
    assert isinstance(model, Model)


def test_model_from_buffer_valid():
    core = Core()
    with open(test_net_bin, "rb") as f:
        weights = f.read()
    with open(model_path()[0], "rb") as f:
        xml = f.read()
    model = core.read_model(model=xml, weights=weights)
    ref_model = core.read_model(model=test_net_xml, weights=test_net_bin)
    assert compare_models(model, ref_model)


def test_get_version(device):
    core = Core()
    version = core.get_versions(device)
    assert isinstance(version, dict), "Returned version must be a dictionary"
    assert device in version, f"{device} plugin version wasn't found in versions"
    assert hasattr(version[device], "major"), "Returned version has no field 'major'"
    assert hasattr(version[device], "minor"), "Returned version has no field 'minor'"
    assert hasattr(version[device], "description"), "Returned version has no field 'description'"
    assert hasattr(version[device], "build_number"), "Returned version has no field 'build_number'"


def test_available_devices(device):
    core = Core()
    devices_attr = core.available_devices
    devices_method = core.get_available_devices()
    for devices in (devices_attr, devices_method):
        assert device in devices, (
            f"Current device '{device}' is not listed in "
            f"available devices '{', '.join(devices)}'"
        )


def test_get_property(device):
    core = Core()
    conf = core.get_property(device, "SUPPORTED_CONFIG_KEYS")
    assert "PERF_COUNT" in conf


@pytest.mark.skipif(
    os.environ.get("TEST_DEVICE", "CPU") != "CPU",
    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test",
)
def test_get_property_list_of_str():
    core = Core()
    param = core.get_property("CPU", "OPTIMIZATION_CAPABILITIES")
    assert isinstance(param, list), (
        "Parameter value for 'OPTIMIZATION_CAPABILITIES' "
        f"metric must be a list but {type(param)} is returned"
    )
    assert all(
        isinstance(v, str) for v in param
    ), "Not all of the parameter values for 'OPTIMIZATION_CAPABILITIES' metric are strings!"


@pytest.mark.skipif(
    os.environ.get("TEST_DEVICE", "CPU") != "CPU",
    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test",
)
def test_get_property_tuple_of_two_ints():
    core = Core()
    param = core.get_property("CPU", "RANGE_FOR_STREAMS")
    assert isinstance(param, tuple), (
        "Parameter value for 'RANGE_FOR_STREAMS' "
        f"metric must be tuple but {type(param)} is returned"
    )
    assert all(
        isinstance(v, int) for v in param
    ), "Not all of the parameter values for 'RANGE_FOR_STREAMS' metric are integers!"


@pytest.mark.skipif(
    os.environ.get("TEST_DEVICE", "CPU") != "CPU",
    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test",
)
def test_get_property_tuple_of_three_ints():
    core = Core()
    param = core.get_property("CPU", "RANGE_FOR_ASYNC_INFER_REQUESTS")
    assert isinstance(param, tuple), (
        "Parameter value for 'RANGE_FOR_ASYNC_INFER_REQUESTS' "
        f"metric must be tuple but {type(param)} is returned"
    )
    assert all(isinstance(v, int) for v in param), (
        "Not all of the parameter values for "
        "'RANGE_FOR_ASYNC_INFER_REQUESTS' metric are integers!"
    )


@pytest.mark.skipif(
    os.environ.get("TEST_DEVICE", "CPU") != "CPU",
    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test",
)
def test_get_property_str():
    core = Core()
    param = core.get_property("CPU", "FULL_DEVICE_NAME")
    assert isinstance(param, str), (
        "Parameter value for 'FULL_DEVICE_NAME' "
        f"metric must be string but {type(param)} is returned"
    )


def test_query_model(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    query_model = core.query_model(model=model, device_name=device)
    ops_model = model.get_ordered_ops()
    ops_func_names = [op.friendly_name for op in ops_model]
    assert [
        key for key in query_model.keys() if key not in ops_func_names
    ] == [], "Not all network layers present in query_model results"
    assert device in next(iter(set(query_model.values()))), "Wrong device for some layers"


@pytest.mark.dynamic_library()
def test_register_plugin():
    device = "TEST_DEVICE"
    lib_name = "test_plugin"
    full_lib_name = lib_name + ".dll" if sys.platform == "win32" else "lib" + lib_name + ".so"

    core = Core()
    core.register_plugin(lib_name, device)
    with pytest.raises(RuntimeError) as e:
        core.get_versions(device)
    assert f"Cannot load library '{full_lib_name}'" in str(e.value)


@pytest.mark.dynamic_library()
def test_register_plugins():
    device = "TEST_DEVICE"
    lib_name = "test_plugin"
    full_lib_name = lib_name + ".dll" if sys.platform == "win32" else "lib" + lib_name + ".so"
    plugins_xml = plugins_path(device, full_lib_name)

    core = Core()
    core.register_plugins(plugins_xml)
    os.remove(plugins_xml)

    with pytest.raises(RuntimeError) as e:
        core.get_versions(device)
    assert f"Cannot load library '{full_lib_name}'" in str(e.value)


def test_unload_plugin(device):
    core = Core()
    # Trigger plugin loading
    core.get_versions(device)
    # Unload plugin
    core.unload_plugin(device)


@pytest.mark.template_extension()
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
    compiled_model = core.compile_model(model, device)
    assert compiled_model.input().partial_shape == after_reshape


def test_add_extension():
    class EmptyExtension(Extension):
        def __init__(self) -> None:
            super().__init__()

    core = Core()
    core.add_extension(EmptyExtension())
    core.add_extension([EmptyExtension(), EmptyExtension()])
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    assert isinstance(model, Model)


def test_read_model_from_buffer_no_weights():
    bytes_model = bytes(
        b"""<net name="add_model" version="10">
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
    model = core.read_model(model=bytes_model)
    assert isinstance(model, Model)


def test_infer_new_request_return_type(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    img = generate_image()
    compiled_model = core.compile_model(model, device)
    res = compiled_model.infer_new_request({"data": img})
    arr = res[list(res)[0]][0]

    assert isinstance(arr, np.ndarray)
    assert arr.itemsize == 4
    assert arr.shape == (10,)
    assert arr.dtype == "float32"
    assert arr.nbytes == 40
