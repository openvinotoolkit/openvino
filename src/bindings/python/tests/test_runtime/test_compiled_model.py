# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import numpy as np

from tests.utils.helpers import (
    get_relu_model,
    generate_image,
    generate_model_and_image,
    generate_concat_compiled_model,
    generate_relu_compiled_model,
    generate_relu_compiled_model_with_config,
    encrypt_base64,
    decrypt_base64,
    create_filenames_for_ir,
    create_filename_for_test)
from openvino import Model, Shape, Core, Tensor, serialize
from openvino import ConstOutput

import openvino.properties as props


def test_get_property(device):
    model = get_relu_model([1, 3, 32, 32])
    core = Core()
    compiled_model = core.compile_model(model, device, {})
    network_name = compiled_model.get_property(props.model_name)
    assert network_name == "test_model"


def test_get_runtime_model(device):
    compiled_model = generate_relu_compiled_model(device)
    runtime_model = compiled_model.get_runtime_model()
    assert isinstance(runtime_model, Model)


def test_export_import(device):
    core = Core()

    if props.device.Capability.EXPORT_IMPORT not in core.get_property(device, props.device.capabilities):
        pytest.skip(f"{core.get_property(device, props.device.full_name)} plugin due-to export, import model API isn't implemented.")

    compiled_model = generate_relu_compiled_model(device)

    user_stream = compiled_model.export_model()

    new_compiled = core.import_model(user_stream, device)

    img = generate_image()
    res = new_compiled.infer_new_request({"data": img})

    assert np.argmax(res[new_compiled.outputs[0]]) == 531


def test_export_import_with_encryption(device):
    core = Core()

    if props.device.Capability.EXPORT_IMPORT not in core.get_property(device, props.device.capabilities):
        pytest.skip(f"{core.get_property(device, props.device.full_name)} plugin due-to export, import model API isn't implemented.")

    config = {}
    config["CACHE_ENCRYPTION_CALLBACKS"] = [encrypt_base64, decrypt_base64]

    compiled_model = generate_relu_compiled_model_with_config(device, config)

    user_stream = compiled_model.export_model()

    new_compiled = core.import_model(user_stream, device, config)

    img = generate_image()
    res = new_compiled.infer_new_request({"data": img})

    assert np.argmax(res[new_compiled.outputs[0]]) == 531


def test_export_import_advanced(device):
    import io

    core = Core()

    if props.device.Capability.EXPORT_IMPORT not in core.get_property(device, props.device.capabilities):
        pytest.skip(f"{core.get_property(device, props.device.full_name)} plugin due-to export, import model API isn't implemented.")

    compiled_model = generate_relu_compiled_model(device)

    user_stream = io.BytesIO()

    compiled_model.export_model(user_stream)

    new_compiled = core.import_model(user_stream, device)

    img = generate_image()
    res = new_compiled.infer_new_request({"data": img})

    assert np.argmax(res[new_compiled.outputs[0]]) == 531


# request - https://docs.pytest.org/en/7.1.x/reference/reference.html#request
@pytest.fixture
def prepare_blob_path(request, tmp_path):
    filename = create_filename_for_test(request.node.name)
    path_to_blob = tmp_path / str(filename + ".blob")
    yield path_to_blob

    os.remove(path_to_blob)


def test_export_import_via_file(prepare_blob_path, device):
    import io

    core = Core()

    if props.device.Capability.EXPORT_IMPORT not in core.get_property(device, props.device.capabilities):
        pytest.skip(f"{core.get_property(device, props.device.full_name)} plugin due-to export, import model API isn't implemented.")

    compiled_model = generate_relu_compiled_model(device)

    user_stream = io.BytesIO()

    compiled_model.export_model(user_stream)
    path_to_blob = prepare_blob_path

    with open(path_to_blob, "wb") as f_w:
        f_w.write(user_stream.getbuffer())

    with open(path_to_blob, "rb") as f_r:
        new_compiled = core.import_model(f_r.read(), device)

    img = generate_image()
    res = new_compiled.infer_new_request({"data": img})

    assert np.argmax(res[new_compiled.outputs[0]]) == 531


@pytest.mark.parametrize("input_arguments", [[0], ["data"], []])
def test_get_input(device, input_arguments):
    compiled_model = generate_relu_compiled_model(device)
    net_input = compiled_model.input(*input_arguments)
    assert isinstance(net_input, ConstOutput)
    assert net_input.get_node().friendly_name == "data"


@pytest.mark.parametrize("output_arguments", [[0], []])
def test_get_output(device, output_arguments):
    compiled_model = generate_relu_compiled_model(device)
    output = compiled_model.output(*output_arguments)
    assert isinstance(output, ConstOutput)


def test_input_set_friendly_name(device):
    compiled_model = generate_relu_compiled_model(device)
    net_input = compiled_model.input("data")
    input_node = net_input.get_node()
    input_node.set_friendly_name("input_1")
    name = input_node.friendly_name
    assert isinstance(net_input, ConstOutput)
    assert name == "input_1"


def test_output_set_friendly_name(device):
    compiled_model = generate_relu_compiled_model(device)
    output = compiled_model.output(0)
    output_node = output.get_node()
    output_node.set_friendly_name("output_1")
    name = output_node.friendly_name
    assert isinstance(output, ConstOutput)
    assert name == "output_1"


def test_outputs(device):
    compiled_model = generate_relu_compiled_model(device)
    outputs = compiled_model.outputs
    assert isinstance(outputs, list)
    assert len(outputs) == 1
    assert isinstance(outputs[0], ConstOutput)


def test_output_type(device):
    compiled_model = generate_relu_compiled_model(device)
    output = compiled_model.output(0)
    output_type = output.get_element_type().get_type_name()
    assert output_type == "f32"


def test_output_shape(device):
    compiled_model = generate_relu_compiled_model(device)
    output = compiled_model.output(0)
    expected_shape = Shape([1, 3, 32, 32])
    assert str(output.get_shape()) == str(expected_shape)


def test_input_get_index(device):
    compiled_model = generate_relu_compiled_model(device)
    net_input = compiled_model.input(0)
    assert net_input.get_index() == 0


def test_inputs(device):
    compiled_model = generate_relu_compiled_model(device)
    inputs = compiled_model.inputs
    assert isinstance(inputs, list)
    assert len(inputs) == 1
    assert isinstance(inputs[0], ConstOutput)


def test_inputs_get_friendly_name(device):
    compiled_model = generate_relu_compiled_model(device)
    node = compiled_model.inputs[0].get_node()
    name = node.friendly_name
    assert name == "data"


def test_inputs_set_friendly_name(device):
    compiled_model = generate_relu_compiled_model(device)
    node = compiled_model.inputs[0].get_node()
    node.set_friendly_name("input_0")
    name = node.friendly_name
    assert name == "input_0"


def test_inputs_docs(device):
    compiled_model = generate_relu_compiled_model(device)

    input_0 = compiled_model.inputs[0]
    assert input_0.__doc__ == "openvino.ConstOutput represents port/node output."


def test_infer_new_request_numpy(device):
    compiled_model, img = generate_model_and_image(device)
    res = compiled_model.infer_new_request({"data": img})
    assert np.argmax(res[list(res)[0]]) == 531


def test_infer_new_request_tensor_numpy_copy(device):
    compiled_model, img = generate_model_and_image(device)

    tensor = Tensor(img)
    res_tensor = compiled_model.infer_new_request({"data": tensor})
    res_img = compiled_model.infer_new_request({"data": img})
    assert np.argmax(res_tensor[list(res_tensor)[0]]) == 531
    assert np.argmax(res_tensor[list(res_tensor)[0]]) == np.argmax(res_img[list(res_img)[0]])


def test_infer_tensor_numpy_shared_memory(device):
    compiled_model, img = generate_model_and_image(device)

    img = np.ascontiguousarray(img)
    tensor = Tensor(img, shared_memory=True)
    res_tensor = compiled_model.infer_new_request({"data": tensor})
    res_img = compiled_model.infer_new_request({"data": img})
    assert np.argmax(res_tensor[list(res_tensor)[0]]) == 531
    assert np.argmax(res_tensor[list(res_tensor)[0]]) == np.argmax(res_img[list(res_img)[0]])


def test_infer_new_request_wrong_port_name(device):
    compiled_model, img = generate_model_and_image(device)

    tensor = Tensor(img)
    with pytest.raises(RuntimeError) as e:
        compiled_model.infer_new_request({"_data_": tensor})
    assert "Check" in str(e.value)


def test_infer_tensor_wrong_input_data(device):
    compiled_model, img = generate_model_and_image(device)

    img = np.ascontiguousarray(img)
    tensor = Tensor(img, shared_memory=True)
    with pytest.raises(TypeError) as e:
        compiled_model.infer_new_request({0.: tensor})
    assert "Incompatible key type for input: 0.0" in str(e.value)


@pytest.mark.parametrize("shared_flag", [True, False])
def test_direct_infer(device, shared_flag):
    compiled_model, img = generate_model_and_image(device)

    tensor = Tensor(img)
    res = compiled_model({"data": tensor}, share_inputs=shared_flag)
    assert np.argmax(res[compiled_model.outputs[0]]) == 531
    ref = compiled_model.infer_new_request({"data": tensor})
    assert np.array_equal(ref[compiled_model.outputs[0]], res[compiled_model.outputs[0]])


# request - https://docs.pytest.org/en/7.1.x/reference/reference.html#request
def test_compiled_model_after_core_destroyed(request, tmp_path, device):
    core = Core()
    xml_path, bin_path = create_filenames_for_ir(request.node.name, tmp_path)
    model = get_relu_model()
    serialize(model, xml_path, bin_path)
    with open(bin_path, "rb") as f:
        weights = f.read()
    with open(xml_path, "rb") as f:
        xml = f.read()
    model = core.read_model(model=xml, weights=weights)
    compiled = core.compile_model(model, device)
    del core
    del model
    # check compiled and infer request can work properly after core object is destroyed
    compiled([np.random.normal(size=list(input.shape)).astype(dtype=input.get_element_type().to_dtype()) for input in compiled.inputs])


def test_compiled_model_from_buffer_in_memory(request, tmp_path, device):
    core = Core()
    xml_path, bin_path = create_filenames_for_ir(request.node.name, tmp_path)
    model = get_relu_model()
    serialize(model, xml_path, bin_path)
    with open(bin_path, "rb") as f:
        weights = f.read()
    with open(xml_path, "r") as f:
        xml = f.read()

    compiled = core.compile_model(model=xml, weights=weights, device_name=device)
    assert isinstance(compiled.outputs[0], ConstOutput)
    _ = compiled([np.random.normal(size=list(input.shape)).astype(dtype=input.get_element_type().to_dtype()) for input in compiled.inputs])


def test_memory_release(device):
    compiled_model = generate_concat_compiled_model(device)
    request = compiled_model.create_infer_request()

    input_tensor = Tensor(compiled_model.inputs[0].get_element_type(), compiled_model.inputs[0].get_shape())
    request.infer({0: input_tensor, 1: input_tensor})

    # Release memory and perform inference again
    compiled_model.release_memory()
    request.infer({0: input_tensor, 1: input_tensor})
