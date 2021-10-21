import numpy as np
import os
import pytest
import warnings
import time
from pathlib import Path

from ..conftest import model_path, image_path
from openvino.impl import Function, ConstOutput, Shape, PartialShape

from openvino import Core

is_myriad = os.environ.get("TEST_DEVICE") == "MYRIAD"
path_to_image = image_path()
test_net_xml, test_net_bin = model_path(is_myriad)

def image_path():
    path_to_repo = os.environ["DATA_PATH"]
    path_to_img = os.path.join(path_to_repo, "validation_set", "224x224", "dog.bmp")
    return path_to_img


def model_path(is_myriad=False):
    path_to_repo = os.environ["MODELS_PATH"]
    if not is_myriad:
        test_xml = os.path.join(path_to_repo, "models", "test_model", "test_model_fp32.xml")
        test_bin = os.path.join(path_to_repo, "models", "test_model", "test_model_fp32.bin")
    else:
        test_xml = os.path.join(path_to_repo, "models", "test_model", "test_model_fp16.xml")
        test_bin = os.path.join(path_to_repo, "models", "test_model", "test_model_fp16.bin")
    return (test_xml, test_bin)


def read_image():
    import cv2
    n, c, h, w = (1, 3, 32, 32)
    image = cv2.imread(path_to_img)
    if image is None:
        raise FileNotFoundError("Input image not found")

    image = cv2.resize(image, (h, w)) / 255
    image = image.transpose((2, 0, 1)).astype(np.float32)
    image = image.reshape((n, c, h, w))
    return image


def test_get_runtime_function(device):
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(func, device)
    runtime_func = exec_net.get_runtime_function()
    assert isinstance(runtime_func, Function)


def test_get_input_i(device):
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(func, device)
    input = exec_net.input(0);
    input_node = input.get_node()
    name = input_node.friendly_name
    assert isinstance(input, ConstOutput)
    assert name == "data"


def test_get_input_tensor_name(device):
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(func, device)
    input = exec_net.input("data");
    input_node = input.get_node()
    name = input_node.friendly_name
    assert isinstance(input, ConstOutput)
    assert name == "data"


def test_get_input(device):
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(func, device)
    input = exec_net.input();
    input_node = input.get_node()
    name = input_node.friendly_name
    assert isinstance(input, ConstOutput)
    assert name == "data"


def test_get_output_i(device):
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(func, device)
    output = exec_net.output(0);
    assert isinstance(output, ConstOutput)


def test_get_output(device):
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(func, device)
    output = exec_net.output();
    output_node = output.get_node()
    assert isinstance(output, ConstOutput)


def test_input_set_friendly_name(device):
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(func, device)
    input = exec_net.input("data");
    input_node = input.get_node()
    input_node.set_friendly_name("input_1")
    name = input_node.friendly_name
    assert isinstance(input, ConstOutput)
    assert name == "input_1"


def test_output_set_friendly_name(device):
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(func, device)
    output = exec_net.output(0);
    output_node = output.get_node()
    output_node.set_friendly_name("output_1")
    name = output_node.friendly_name
    assert isinstance(output, ConstOutput)
    assert name == "output_1"


def test_outputs(device):
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(func, device)
    outputs = exec_net.outputs
    assert isinstance(outputs, list)
    assert len(outputs) == 1


def test_outputs_items(device):
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(func, device)
    outputs = exec_net.outputs
    assert isinstance(outputs[0], ConstOutput)


def test_output_type(device):
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(func, device)
    output = exec_net.output(0)
    output_type = output.get_element_type().get_type_name()
    assert output_type == "f32"

def test_output_shape(device):
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(func, device)
    output = exec_net.output(0)
    expected_shape = Shape([1, 10])
    assert str(output.get_shape()) == str(expected_shape)


def test_input_get_index(device):
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(func, device)
    input = exec_net.input(0)
    expected_idx = 0
    assert input.get_index() == expected_idx


def test_input_get_index(device):
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(func, device)
    input = exec_net.input(0)
    expected_partial_shape = PartialShape([1, 3, 32 ,32])
    assert input.get_partial_shape() == expected_partial_shape
    

def test_inputs(device):
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(func, device)
    inputs = exec_net.inputs
    assert isinstance(inputs, list)
    assert len(inputs) == 1


def test_inputs_items(device):
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(func, device)
    inputs = exec_net.inputs
    assert isinstance(inputs[0], ConstOutput)


def test_inputs_get_friendly_name(device):
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(func, device)
    inputs = exec_net.inputs
    input_0 = inputs[0]
    node = input_0.get_node()
    name = node.friendly_name
    assert name == "data"


def test_inputs_set_friendly_name(device):
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(func, device)
    inputs = exec_net.inputs
    input_0 = inputs[0]
    node = input_0.get_node()
    node.set_friendly_name("input_0")
    name = node.friendly_name
    assert name == "input_0"


def test_inputs_docs(device):
    ie = Core()
    func = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(func, device)
    inputs = exec_net.inputs
    input_0 = inputs[0]
    exptected_string = "openvino.impl.ConstOutput wraps ov::Output<Const ov::Node >"
    assert input_0.__doc__ == exptected_string

