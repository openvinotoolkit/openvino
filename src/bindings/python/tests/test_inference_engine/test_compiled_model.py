# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import numpy as np

# TODO: refactor into absolute paths
from ..conftest import model_path, get_model_with_template_extension
from ..test_utils.test_utils import generate_image
from openvino.runtime import Model, ConstOutput, Shape

from openvino.runtime import Core, Tensor

is_myriad = os.environ.get("TEST_DEVICE") == "MYRIAD"
test_net_xml, test_net_bin = model_path(is_myriad)


def test_get_property_model_name(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    network_name = exec_net.get_property("NETWORK_NAME")
    assert network_name == "test_model"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU", reason="Device dependent test")
def test_get_property(device):
    core = Core()
    if core.get_property(device, "FULL_DEVICE_NAME") == "arm_compute::NEON":
        pytest.skip("Can't run on ARM plugin due-to CPU dependent test")
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    profiling_enabled = exec_net.get_property("PERF_COUNT")
    assert not profiling_enabled


def test_get_runtime_model(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    runtime_func = exec_net.get_runtime_model()
    assert isinstance(runtime_func, Model)


def test_export_import():
    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled = core.compile_model(model, "CPU")

    user_stream = compiled.export_model()

    new_compiled = core.import_model(user_stream, "CPU")

    img = generate_image()
    res = new_compiled.infer_new_request({"data": img})

    assert np.argmax(res[new_compiled.outputs[0]]) == 9


def test_export_import_advanced():
    import io

    core = Core()
    model = core.read_model(model=test_net_xml, weights=test_net_bin)
    compiled = core.compile_model(model, "CPU")

    user_stream = io.BytesIO()

    compiled.export_model(user_stream)

    new_compiled = core.import_model(user_stream, "CPU")

    img = generate_image()
    res = new_compiled.infer_new_request({"data": img})

    assert np.argmax(res[new_compiled.outputs[0]]) == 9


def test_get_input_i(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    net_input = exec_net.input(0)
    input_node = net_input.get_node()
    name = input_node.friendly_name
    assert isinstance(net_input, ConstOutput)
    assert name == "data"


def test_get_input_tensor_name(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    net_input = exec_net.input("data")
    input_node = net_input.get_node()
    name = input_node.friendly_name
    assert isinstance(net_input, ConstOutput)
    assert name == "data"


def test_get_input(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    net_input = exec_net.input()
    input_node = net_input.get_node()
    name = input_node.friendly_name
    assert isinstance(net_input, ConstOutput)
    assert name == "data"


def test_get_output_i(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    output = exec_net.output(0)
    assert isinstance(output, ConstOutput)


def test_get_output(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    output = exec_net.output()
    assert isinstance(output, ConstOutput)


def test_input_set_friendly_name(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    net_input = exec_net.input("data")
    input_node = net_input.get_node()
    input_node.set_friendly_name("input_1")
    name = input_node.friendly_name
    assert isinstance(net_input, ConstOutput)
    assert name == "input_1"


def test_output_set_friendly_name(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    output = exec_net.output(0)
    output_node = output.get_node()
    output_node.set_friendly_name("output_1")
    name = output_node.friendly_name
    assert isinstance(output, ConstOutput)
    assert name == "output_1"


def test_outputs(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    outputs = exec_net.outputs
    assert isinstance(outputs, list)
    assert len(outputs) == 1


def test_outputs_items(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    outputs = exec_net.outputs
    assert isinstance(outputs[0], ConstOutput)


def test_output_type(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    output = exec_net.output(0)
    output_type = output.get_element_type().get_type_name()
    assert output_type == "f32"


def test_output_shape(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    output = exec_net.output(0)
    expected_shape = Shape([1, 10])
    assert str(output.get_shape()) == str(expected_shape)


def test_input_get_index(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    net_input = exec_net.input(0)
    expected_idx = 0
    assert net_input.get_index() == expected_idx


def test_inputs(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    inputs = exec_net.inputs
    assert isinstance(inputs, list)
    assert len(inputs) == 1


def test_inputs_items(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    inputs = exec_net.inputs
    assert isinstance(inputs[0], ConstOutput)


def test_inputs_get_friendly_name(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    inputs = exec_net.inputs
    input_0 = inputs[0]
    node = input_0.get_node()
    name = node.friendly_name
    assert name == "data"


def test_inputs_set_friendly_name(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    inputs = exec_net.inputs
    input_0 = inputs[0]
    node = input_0.get_node()
    node.set_friendly_name("input_0")
    name = node.friendly_name
    assert name == "input_0"


def test_inputs_docs(device):
    core = Core()
    func = core.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = core.compile_model(func, device)
    inputs = exec_net.inputs
    input_0 = inputs[0]
    expected_string = "openvino.runtime.ConstOutput represents port/node output."
    assert input_0.__doc__ == expected_string


def test_infer_new_request_numpy(device):
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    img = generate_image()
    exec_net = ie.compile_model(func, device)
    res = exec_net.infer_new_request({"data": img})
    assert np.argmax(res[list(res)[0]]) == 9


def test_infer_new_request_tensor_numpy_copy(device):
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    img = generate_image()
    tensor = Tensor(img)
    exec_net = ie.compile_model(func, device)
    res_tensor = exec_net.infer_new_request({"data": tensor})
    res_img = exec_net.infer_new_request({"data": tensor})
    assert np.argmax(res_tensor[list(res_tensor)[0]]) == 9
    assert np.argmax(res_tensor[list(res_tensor)[0]]) == np.argmax(res_img[list(res_img)[0]])


def test_infer_tensor_numpy_shared_memory(device):
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    img = generate_image()
    img = np.ascontiguousarray(img)
    tensor = Tensor(img, shared_memory=True)
    exec_net = ie.compile_model(func, device)
    res_tensor = exec_net.infer_new_request({"data": tensor})
    res_img = exec_net.infer_new_request({"data": tensor})
    assert np.argmax(res_tensor[list(res_tensor)[0]]) == 9
    assert np.argmax(res_tensor[list(res_tensor)[0]]) == np.argmax(res_img[list(res_img)[0]])


def test_infer_new_request_wrong_port_name(device):
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    img = generate_image()
    tensor = Tensor(img)
    exec_net = ie.compile_model(func, device)
    with pytest.raises(RuntimeError) as e:
        exec_net.infer_new_request({"_data_": tensor})
    assert "Check" in str(e.value)


def test_infer_tensor_wrong_input_data(device):
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    img = generate_image()
    img = np.ascontiguousarray(img)
    tensor = Tensor(img, shared_memory=True)
    exec_net = ie.compile_model(func, device)
    with pytest.raises(TypeError) as e:
        exec_net.infer_new_request({0.: tensor})
    assert "Incompatible key type for input: 0.0" in str(e.value)


def test_infer_numpy_model_from_buffer(device):
    core = Core()
    with open(test_net_bin, "rb") as f:
        weights = f.read()
    with open(test_net_xml, "rb") as f:
        xml = f.read()
    func = core.read_model(model=xml, weights=weights)
    img = generate_image()
    exec_net = core.compile_model(func, device)
    res = exec_net.infer_new_request({"data": img})
    assert np.argmax(res[list(res)[0]]) == 9


def test_infer_tensor_model_from_buffer(device):
    core = Core()
    with open(test_net_bin, "rb") as f:
        weights = f.read()
    with open(test_net_xml, "rb") as f:
        xml = f.read()
    func = core.read_model(model=xml, weights=weights)
    img = generate_image()
    tensor = Tensor(img)
    exec_net = core.compile_model(func, device)
    res = exec_net.infer_new_request({"data": tensor})
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


def test_compiled_model_after_core_destroyed(device):
    core, model = get_model_with_template_extension()
    compiled = core.compile_model(model, device)
    del core
    del model
    # check compiled and infer request can work properly after core object is destroyed
    compiled([np.random.normal(size=list(input.shape)) for input in compiled.inputs])
