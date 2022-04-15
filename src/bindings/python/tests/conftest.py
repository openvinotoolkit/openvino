# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest

import tests

from pathlib import Path
from sys import platform
from openvino.runtime import Core


def get_model_with_template_extension():
    core = Core()
    ir = bytes(b"""<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data shape="1,3,22,22" element_type="f32"/>
            <output>
                <port id="0" precision="FP32" names="in_data">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="Identity" version="extension">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32" names="out_data">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>""")
    if platform == "win32":
        core.add_extension(library_path="openvino_template_extension.dll")
    else:
        core.add_extension(library_path="libopenvino_template_extension.so")
    return core, core.read_model(ir)


def model_path(is_myriad=False):
    base_path = os.path.dirname(__file__)
    if not is_myriad:
        test_xml = os.path.join(base_path, "test_utils", "utils", "test_model_fp32.xml")
        test_bin = os.path.join(base_path, "test_utils", "utils", "test_model_fp32.bin")
    else:
        test_xml = os.path.join(base_path, "test_utils", "utils", "test_model_fp16.xml")
        test_bin = os.path.join(base_path, "test_utils", "utils", "test_model_fp16.bin")
    return (test_xml, test_bin)


def model_onnx_path():
    base_path = os.path.dirname(__file__)
    test_onnx = os.path.join(base_path, "test_utils", "utils", "test_model.onnx")
    return test_onnx


def plugins_path():
    base_path = os.path.dirname(__file__)
    plugins_xml = os.path.join(base_path, "test_utils", "utils", "plugins.xml")
    plugins_win_xml = os.path.join(base_path, "test_utils", "utils", "plugins_win.xml")
    plugins_osx_xml = os.path.join(base_path, "test_utils", "utils", "plugins_apple.xml")
    return (plugins_xml, plugins_win_xml, plugins_osx_xml)


def _get_default_model_zoo_dir():
    return Path(os.getenv("ONNX_HOME", Path.home() / ".onnx/model_zoo"))


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        default="CPU",
        choices=["CPU", "GPU", "HDDL", "MYRIAD", "HETERO", "TEMPLATE"],
        help="Select target device",
    )
    parser.addoption(
        "--model_zoo_dir",
        default=_get_default_model_zoo_dir(),
        type=str,
        help="location of the model zoo",
    )
    parser.addoption(
        "--model_zoo_xfail",
        action="store_true",
        help="treat model zoo known issues as xfails instead of failures",
    )


def pytest_configure(config):
    backend_name = config.getvalue("backend")
    tests.BACKEND_NAME = backend_name
    tests.MODEL_ZOO_DIR = Path(config.getvalue("model_zoo_dir"))
    tests.MODEL_ZOO_XFAIL = config.getvalue("model_zoo_xfail")

    # register additional markers
    config.addinivalue_line("markers", "skip_on_cpu: Skip test on CPU")
    config.addinivalue_line("markers", "skip_on_gpu: Skip test on GPU")
    config.addinivalue_line("markers", "skip_on_hddl: Skip test on HDDL")
    config.addinivalue_line("markers", "skip_on_myriad: Skip test on MYRIAD")
    config.addinivalue_line("markers", "skip_on_hetero: Skip test on HETERO")
    config.addinivalue_line("markers", "skip_on_template: Skip test on TEMPLATE")
    config.addinivalue_line("markers", "onnx_coverage: Collect ONNX operator coverage")
    config.addinivalue_line("markers", "template_extension")
    config.addinivalue_line("markers", "dynamic_library: Runs tests only in dynamic libraries case")


def pytest_collection_modifyitems(config, items):
    backend_name = config.getvalue("backend")
    tests.MODEL_ZOO_DIR = Path(config.getvalue("model_zoo_dir"))
    tests.MODEL_ZOO_XFAIL = config.getvalue("model_zoo_xfail")

    keywords = {
        "CPU": "skip_on_cpu",
        "GPU": "skip_on_gpu",
        "HDDL": "skip_on_hddl",
        "MYRIAD": "skip_on_myriad",
        "HETERO": "skip_on_hetero",
        "TEMPLATE": "skip_on_template",
    }

    skip_markers = {
        "CPU": pytest.mark.skip(reason="Skipping test on the CPU backend."),
        "GPU": pytest.mark.skip(reason="Skipping test on the GPU backend."),
        "HDDL": pytest.mark.skip(reason="Skipping test on the HDDL backend."),
        "MYRIAD": pytest.mark.skip(reason="Skipping test on the MYRIAD backend."),
        "HETERO": pytest.mark.skip(reason="Skipping test on the HETERO backend."),
        "TEMPLATE": pytest.mark.skip(reason="Skipping test on the TEMPLATE backend."),
    }

    for item in items:
        skip_this_backend = keywords[backend_name]
        if skip_this_backend in item.keywords:
            item.add_marker(skip_markers[backend_name])


@pytest.fixture(scope="session")
def device():
    return os.environ.get("TEST_DEVICE") if os.environ.get("TEST_DEVICE") else "CPU"
