# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import numpy as np

from tests.conftest import model_path
from tests.test_utils.test_utils import generate_image
from openvino.runtime import Model, ConstOutput, Shape

from openvino.runtime import Core, Tensor

is_myriad = os.environ.get("TEST_DEVICE") == "MYRIAD"
test_net_xml, test_net_bin = model_path(is_myriad)


def test_get_property_model_name(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, device)
    network_name = compiled_model.get_property("NETWORK_NAME")
    assert network_name == "test_model"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU", reason="Device dependent test")
def test_get_property(device):
    core = Core()
    if core.get_property(device, "FULL_DEVICE_NAME") == "arm_compute::NEON":
        pytest.skip("Can't run on ARM plugin due-to CPU dependent test")
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, device)
    profiling_enabled = compiled_model.get_property("PERF_COUNT")
    assert not profiling_enabled


def test_get_runtime_model(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, device)
    runtime_model = compiled_model.get_runtime_model()
    assert isinstance(runtime_model, Model)


def test_export_import():
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, "CPU")

    user_stream = compiled_model.export_model()

    new_compiled = core.import_model(user_stream, "CPU")

    img = generate_image()
    res = new_compiled.infer_new_request({"data": img})

    assert np.argmax(res[new_compiled.outputs[0]]) == 9


def test_export_import_advanced():
    import io

    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, "CPU")

    user_stream = io.BytesIO()

    compiled_model.export_model(user_stream)

    new_compiled = core.import_model(user_stream, "CPU")

    img = generate_image()
    res = new_compiled.infer_new_request({"data": img})

    assert np.argmax(res[new_compiled.outputs[0]]) == 9


def test_get_input_i(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, device)
    net_input = compiled_model.input(0)
    input_node = net_input.get_node()
    name = input_node.friendly_name
    assert isinstance(net_input, ConstOutput)
    assert name == "data"


def test_get_input_tensor_name(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, device)
    net_input = compiled_model.input("data")
    input_node = net_input.get_node()
    name = input_node.friendly_name
    assert isinstance(net_input, ConstOutput)
    assert name == "data"


def test_get_input(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, device)
    net_input = compiled_model.input()
    input_node = net_input.get_node()
    name = input_node.friendly_name
    assert isinstance(net_input, ConstOutput)
    assert name == "data"


def test_get_output_i(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, device)
    output = compiled_model.output(0)
    assert isinstance(output, ConstOutput)


def test_get_output(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, device)
    output = compiled_model.output()
    assert isinstance(output, ConstOutput)


def test_input_set_friendly_name(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, device)
    net_input = compiled_model.input("data")
    input_node = net_input.get_node()
    input_node.set_friendly_name("input_1")
    name = input_node.friendly_name
    assert isinstance(net_input, ConstOutput)
    assert name == "input_1"


def test_output_set_friendly_name(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, device)
    output = compiled_model.output(0)
    output_node = output.get_node()
    output_node.set_friendly_name("output_1")
    name = output_node.friendly_name
    assert isinstance(output, ConstOutput)
    assert name == "output_1"


def test_outputs(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, device)
    outputs = compiled_model.outputs
    assert isinstance(outputs, list)
    assert len(outputs) == 1


def test_outputs_items(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, device)
    outputs = compiled_model.outputs
    assert isinstance(outputs[0], ConstOutput)


def test_output_type(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, device)
    output = compiled_model.output(0)
    output_type = output.get_element_type().get_type_name()
    assert output_type == "f32"


def test_output_shape(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, device)
    output = compiled_model.output(0)
    expected_shape = Shape([1, 10])
    assert str(output.get_shape()) == str(expected_shape)


def test_input_get_index(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, device)
    net_input = compiled_model.input(0)
    expected_idx = 0
    assert net_input.get_index() == expected_idx


def test_inputs(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, device)
    inputs = compiled_model.inputs
    assert isinstance(inputs, list)
    assert len(inputs) == 1


def test_inputs_items(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, device)
    inputs = compiled_model.inputs
    assert isinstance(inputs[0], ConstOutput)


def test_inputs_get_friendly_name(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, device)
    inputs = compiled_model.inputs
    input_0 = inputs[0]
    node = input_0.get_node()
    name = node.friendly_name
    assert name == "data"


def test_inputs_set_friendly_name(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, device)
    inputs = compiled_model.inputs
    input_0 = inputs[0]
    node = input_0.get_node()
    node.set_friendly_name("input_0")
    name = node.friendly_name
    assert name == "input_0"


def test_inputs_docs(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled_model = core.compile_model(model, device)
    inputs = compiled_model.inputs
    input_0 = inputs[0]
    expected_string = "openvino.runtime.ConstOutput represents port/node output."
    assert input_0.__doc__ == expected_string


def test_infer_new_request_numpy(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    img = generate_image()
    compiled_model = core.compile_model(model, device)
    res = compiled_model.infer_new_request({"data": img})
    assert np.argmax(res[list(res)[0]]) == 9


def test_infer_new_request_tensor_numpy_copy(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    img = generate_image()
    tensor = Tensor(img)
    compiled_model = core.compile_model(model, device)
    res_tensor = compiled_model.infer_new_request({"data": tensor})
    res_img = compiled_model.infer_new_request({"data": img})
    assert np.argmax(res_tensor[list(res_tensor)[0]]) == 9
    assert np.argmax(res_tensor[list(res_tensor)[0]]) == np.argmax(res_img[list(res_img)[0]])


def test_infer_tensor_numpy_shared_memory(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    img = generate_image()
    img = np.ascontiguousarray(img)
    tensor = Tensor(img, shared_memory=True)
    compiled_model = core.compile_model(model, device)
    res_tensor = compiled_model.infer_new_request({"data": tensor})
    res_img = compiled_model.infer_new_request({"data": img})
    assert np.argmax(res_tensor[list(res_tensor)[0]]) == 9
    assert np.argmax(res_tensor[list(res_tensor)[0]]) == np.argmax(res_img[list(res_img)[0]])


def test_infer_new_request_wrong_port_name(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    img = generate_image()
    tensor = Tensor(img)
    compiled_model = core.compile_model(model, device)
    with pytest.raises(RuntimeError) as e:
        compiled_model.infer_new_request({"_data_": tensor})
    assert "Check" in str(e.value)


def test_infer_tensor_wrong_input_data(device):
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    img = generate_image()
    img = np.ascontiguousarray(img)
    tensor = Tensor(img, shared_memory=True)
    compiled_model = core.compile_model(model, device)
    with pytest.raises(TypeError) as e:
        compiled_model.infer_new_request({0.: tensor})
    assert "Incompatible key type for input: 0.0" in str(e.value)


def test_infer_numpy_model_from_buffer(device):
    core = Core()
    with open(test_net_bin, "rb") as f:
        weights = f.read()
    with open(test_net_xml, "rb") as f:
        xml = f.read()
    model = core.read_model(model=xml, weights=weights)
    img = generate_image()
    compiled_model = core.compile_model(model, device)
    res = compiled_model.infer_new_request({"data": img})
    assert np.argmax(res[list(res)[0]]) == 9


def test_infer_tensor_model_from_buffer(device):
    core = Core()
    with open(test_net_bin, "rb") as f:
        weights = f.read()
    with open(test_net_xml, "rb") as f:
        xml = f.read()
    model = core.read_model(model=xml, weights=weights)
    img = generate_image()
    tensor = Tensor(img)
    compiled_model = core.compile_model(model, device)
    res = compiled_model.infer_new_request({"data": tensor})
    assert np.argmax(res[list(res)[0]]) == 9


def test_direct_infer(device):
    core = Core()
    with open(test_net_bin, "rb") as f:
        weights = f.read()
    with open(test_net_xml, "rb") as f:
        xml = f.read()
    model = core.read_model(model=xml, weights=weights)
    img = generate_image()
    tensor = Tensor(img)
    comp_model = core.compile_model(model, device)
    res = comp_model({"data": tensor})
    assert np.argmax(res[comp_model.outputs[0]]) == 9
    ref = comp_model.infer_new_request({"data": tensor})
    assert np.array_equal(ref[comp_model.outputs[0]], res[comp_model.outputs[0]])


@pytest.mark.template_plugin()
def test_compiled_model_after_core_destroyed(device):
    core = Core()
    with open(test_net_bin, "rb") as f:
        weights = f.read()
    with open(test_net_xml, "rb") as f:
        xml = f.read()
    model = core.read_model(model=xml, weights=weights)
    compiled = core.compile_model(model, device)
    del core
    del model
    # check compiled and infer request can work properly after core object is destroyed
    compiled([np.random.normal(size=list(input.shape)) for input in compiled.inputs])
