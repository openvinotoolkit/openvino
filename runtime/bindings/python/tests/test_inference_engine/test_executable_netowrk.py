import numpy as np
import os
import pytest
import warnings
import time
from pathlib import Path

from ..conftest import model_path, image_path
from openvino.impl import Function, ConstOutput

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


is_myriad = os.environ.get("TEST_DEVICE") == "MYRIAD"
test_net_xml, test_net_bin = model_path(is_myriad)
path_to_img = image_path()

def test_get_runtime_function(device):
    ie = Core()
    net = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(net, device)
    runtime_func = exec_net.get_runtime_function()
    assert isinstance(runtime_func, Function)


def test_get_input_i(device):
    ie = Core()
    net = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(net, device)
    input = exec_net.input(0);
    input_node = input.get_node
    name = input_node.friendly_name
    assert isinstance(input, ConstOutput)
    assert name == "data"


def test_get_input_tensor_name(device):
    ie = Core()
    net = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(net, device)
    input = exec_net.input("data");
    input_node = input.get_node
    name = input_node.friendly_name
    assert isinstance(input, ConstOutput)
    assert name == "data"


def test_get_input(device):
    ie = Core()
    net = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(net, device)
    input = exec_net.input("data");
    input_node = input.get_node
    name = input_node.friendly_name
    assert isinstance(input, ConstOutput)
    assert name == "data"


def test_get_output_i(device):
    ie = Core()
    net = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(net, device)
    output = exec_net.output(0);
    assert isinstance(output, ConstOutput)


def test_input_set_friendly_name(device):
    ie = Core()
    net = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(net, device)
    input = exec_net.input("data");
    input_node = input.get_node
    input_node.set_friendly_name("input_1")
    name = input_node.friendly_name
    assert isinstance(input, ConstOutput)
    assert name == "input_1"

def test_output_set_friendly_name(device):
    ie = Core()
    net = ie.read_model(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.compile_model(net, device)
    output = exec_net.output(0);
    output_node = output.get_node
    output_node.set_friendly_name("output_1")
    name = output_node.friendly_name
    assert isinstance(output, ConstOutput)
    assert name == "output_1"

