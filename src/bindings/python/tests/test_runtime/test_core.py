# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
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
    serialize,
)

import openvino.properties as props
import openvino.properties.hint as hints
from openvino import Extension
from tests.utils.helpers import (
    generate_image,
    generate_relu_compiled_model,
    get_relu_model,
    plugins_path,
    compare_models,
    create_filenames_for_ir,
    get_model_with_template_extension,
)


def test_compact_api_xml():
    img = generate_image()

    compiled_model = compile_model(get_relu_model())
    assert isinstance(compiled_model, CompiledModel)
    results = compiled_model.infer_new_request({"data": img})
    assert np.argmax(results[list(results)[0]]) == 531


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
    # convert node may be introduced by API 2.0, which brings some deviation
    assert np.allclose(results[list(results)[0]], expected_output, 1e-4, 1e-4)


# request - https://docs.pytest.org/en/7.1.x/reference/reference.html#request
@pytest.mark.parametrize("device_name", [
    None,
    "CPU",
])
def test_compile_model(request, tmp_path, device_name):
    core = Core()
    xml_path, bin_path = create_filenames_for_ir(request.node.name, tmp_path)
    relu_model = get_relu_model()
    serialize(relu_model, xml_path, bin_path)
    model = core.read_model(model=xml_path, weights=bin_path)
    compiled_model = None
    if device_name is None:
        compiled_model = core.compile_model(model)
    else:
        compiled_model = core.compile_model(model, device_name)

    assert isinstance(compiled_model, CompiledModel)


@pytest.fixture
def get_model():
    return get_relu_model()


@pytest.fixture
def get_model_path(request, tmp_path):
    xml_path, _ = create_filenames_for_ir(request.node.name, tmp_path, True)
    serialize(get_relu_model(), xml_path)
    return Path(xml_path)


@pytest.mark.parametrize("model_type", [
    "get_model",
    "get_model_path",
])
@pytest.mark.parametrize("device_name", [
    None,
    "CPU",
])
@pytest.mark.parametrize("config", [
    None,
    {hints.performance_mode(): hints.PerformanceMode.THROUGHPUT},
    {hints.execution_mode: hints.ExecutionMode.PERFORMANCE},
])
def test_compact_api(model_type, device_name, config, request):
    compiled_model = None

    model = request.getfixturevalue(model_type)
    if device_name is not None:
        compiled_model = compile_model(model=model, device_name=device_name, config=config)
    else:
        compiled_model = compile_model(model=model, config=config)

    assert isinstance(compiled_model, CompiledModel)


# request - https://docs.pytest.org/en/7.1.x/reference/reference.html#request
def test_read_model_from_ir(request, tmp_path):
    core = Core()
    xml_path, bin_path = create_filenames_for_ir(request.node.name, tmp_path)
    relu_model = get_relu_model()
    serialize(relu_model, xml_path, bin_path)
    model = core.read_model(model=xml_path, weights=bin_path)
    assert isinstance(model, Model)

    model = core.read_model(model=xml_path)
    assert isinstance(model, Model)


# request - https://docs.pytest.org/en/7.1.x/reference/reference.html#request
def test_read_model_from_ir_with_user_config(request, tmp_path):
    core = Core()
    xml_path, bin_path = create_filenames_for_ir(request.node.name, tmp_path)
    relu_model = get_relu_model()
    serialize(relu_model, xml_path, bin_path)

    core_cache_dir = core.get_property("CACHE_DIR")
    cache_path = tmp_path / Path("cache")

    model = core.read_model(xml_path, bin_path, config={"CACHE_DIR": f"{cache_path}"})

    assert isinstance(model, Model)
    assert core_cache_dir == core.get_property("CACHE_DIR")
    assert os.path.exists(cache_path)
    os.rmdir(cache_path)


# request - https://docs.pytest.org/en/7.1.x/reference/reference.html#request
def test_read_model_from_tensor(request, tmp_path):
    core = Core()
    xml_path, bin_path = create_filenames_for_ir(request.node.name, tmp_path, is_xml_path=True, is_bin_path=True)
    relu_model = get_relu_model()
    serialize(relu_model, xml_path, bin_path)
    arr = np.ones(shape=(10), dtype=np.int8)
    arr.tofile(bin_path)
    model = open(xml_path).read()
    tensor = tensor_from_file(bin_path)
    model = core.read_model(model=model, weights=tensor)
    assert isinstance(model, Model)


def test_read_model_with_wrong_input():
    core = Core()
    with pytest.raises(TypeError) as e:
        core.read_model(model=3, weights=3)
    assert "Provided python object type <class 'int'> isn't supported as 'model' argument." in str(e.value)


# request - https://docs.pytest.org/en/7.1.x/reference/reference.html#request
def test_read_model_as_path(request, tmp_path):
    core = Core()
    xml_path, bin_path = create_filenames_for_ir(request.node.name, tmp_path, True, True)
    relu_model = get_relu_model()
    serialize(relu_model, xml_path, bin_path)

    model = core.read_model(model=Path(xml_path), weights=Path(bin_path))
    assert isinstance(model, Model)

    model = core.read_model(model=xml_path, weights=Path(bin_path))
    assert isinstance(model, Model)

    model = core.read_model(model=Path(xml_path))
    assert isinstance(model, Model)


# request - https://docs.pytest.org/en/7.1.x/reference/reference.html#request
def test_read_model_as_path_with_user_config(request, tmp_path):
    core = Core()
    xml_path, bin_path = create_filenames_for_ir(request.node.name, tmp_path)
    relu_model = get_relu_model()
    serialize(relu_model, xml_path, bin_path)

    core_cache_dir = core.get_property("CACHE_DIR")
    cache_path = tmp_path / Path("cache_as_path")

    model = core.read_model(Path(xml_path), Path(bin_path), config={"CACHE_DIR": f"{cache_path}"})

    assert isinstance(model, Model)
    assert core_cache_dir == core.get_property("CACHE_DIR")
    assert os.path.exists(cache_path)
    os.rmdir(cache_path)


# request - https://docs.pytest.org/en/7.1.x/reference/reference.html#request
def test_read_model_from_buffer(request, tmp_path):
    core = Core()
    xml_path, bin_path = create_filenames_for_ir(request.node.name, tmp_path)
    relu_model = get_relu_model()
    serialize(relu_model, xml_path, bin_path)
    with open(bin_path, "rb") as f:
        weights = f.read()
    with open(xml_path, "rb") as f:
        xml = f.read()
    model = core.read_model(model=xml, weights=weights)
    assert isinstance(model, Model)


# request - https://docs.pytest.org/en/7.1.x/reference/reference.html#request
def test_read_model_from_bytesio(request, tmp_path):
    from io import BytesIO

    core = Core()
    xml_path, bin_path = create_filenames_for_ir(request.node.name, tmp_path)
    relu_model = get_relu_model()
    serialize(relu_model, xml_path, bin_path)
    with open(bin_path, "rb") as f:
        weights = BytesIO(f.read())
    with open(xml_path, "rb") as f:
        xml = BytesIO(f.read())
    model = core.read_model(model=xml, weights=weights)
    assert isinstance(model, Model)


# request - https://docs.pytest.org/en/7.1.x/reference/reference.html#request
def test_model_from_buffer_valid(request, tmp_path):
    core = Core()
    xml_path, bin_path = create_filenames_for_ir(request.node.name, tmp_path)
    relu_model = get_relu_model()
    serialize(relu_model, xml_path, bin_path)
    with open(bin_path, "rb") as f:
        weights = f.read()
    with open(xml_path, "rb") as f:
        xml = f.read()
    model = core.read_model(model=xml, weights=weights)
    ref_model = core.read_model(model=xml_path, weights=bin_path)
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
    conf = core.get_property(device, props.supported_properties())
    assert props.enable_profiling in conf


@pytest.mark.skipif(
    os.environ.get("TEST_DEVICE", "CPU") != "CPU",
    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test",
)
def test_get_property_list_of_str():
    core = Core()
    param = core.get_property("CPU", props.device.capabilities)
    assert isinstance(param, list), (
        f"Parameter value for {props.device.capabilities} "
        f"metric must be a list but {type(param)} is returned"
    )
    assert all(
        isinstance(v, str) for v in param
    ), f"Not all of the parameter values for {props.device.capabilities} metric are strings!"


@pytest.mark.skipif(
    os.environ.get("TEST_DEVICE", "CPU") != "CPU",
    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test",
)
def test_get_property_tuple_of_two_ints():
    core = Core()
    param = core.get_property("CPU", props.range_for_streams)
    assert isinstance(param, tuple), (
        f"Parameter value for {props.range_for_streams} "
        f"metric must be tuple but {type(param)} is returned"
    )
    assert all(
        isinstance(v, int) for v in param
    ), f"Not all of the parameter values for {props.range_for_stream}s metric are integers!"


@pytest.mark.skipif(
    os.environ.get("TEST_DEVICE", "CPU") != "CPU",
    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test",
)
def test_get_property_tuple_of_three_ints():
    core = Core()
    param = core.get_property("CPU", props.range_for_async_infer_requests)
    assert isinstance(param, tuple), (
        f"Parameter value for {props.range_for_async_infer_requests} "
        f"metric must be tuple but {type(param)} is returned"
    )
    assert all(isinstance(v, int) for v in param), (
        "Not all of the parameter values for "
        f"{props.range_for_async_infer_requests} metric are integers!"
    )


@pytest.mark.skipif(
    os.environ.get("TEST_DEVICE", "CPU") != "CPU",
    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test",
)
def test_get_property_str():
    core = Core()
    param = core.get_property("CPU", props.device.full_name)
    assert isinstance(param, str), (
        f"Parameter value for {props.device.full_name} "
        f"metric must be string but {type(param)} is returned"
    )


def test_query_model(device):
    core = Core()
    model = get_relu_model()
    query_model = core.query_model(model=model, device_name=device)
    ops_model = model.get_ordered_ops()
    ops_model_names = [op.friendly_name for op in ops_model]
    assert [
        key for key in query_model.keys() if key not in ops_model_names
    ] == [], "Not all network layers present in query_model results"
    assert device in next(iter(set(query_model.values()))), "Wrong device for some layers"


@pytest.mark.dynamic_library
def test_register_plugin():
    device = "TEST_DEVICE"
    lib_name = "test_plugin"
    full_lib_name = lib_name + ".dll" if sys.platform == "win32" else "lib" + lib_name + ".so"

    core = Core()
    core.register_plugin(lib_name, device)
    with pytest.raises(RuntimeError) as e:
        core.get_versions(device)
    assert f"Cannot load library '{full_lib_name}'" in str(e.value)


@pytest.mark.dynamic_library
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


@pytest.mark.template_extension
@pytest.mark.dynamic_library
@pytest.mark.xfail(condition=sys.platform == "darwin", reason="Ticket - 132696")
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
    model = get_relu_model()
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
    model = get_relu_model()
    img = generate_image()
    compiled_model = core.compile_model(model, device)
    res = compiled_model.infer_new_request({"data": img})
    arr = res[list(res)[0]][0]

    assert isinstance(arr, np.ndarray)
    assert arr.itemsize == 4
    assert arr.shape == (3, 32, 32)
    assert arr.dtype == "float32"
    assert arr.nbytes == 12288
