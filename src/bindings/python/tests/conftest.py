# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest

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


def model_path(is_fp16=False):
    base_path = os.path.dirname(__file__)
    if is_fp16:
        test_xml = os.path.join(base_path, "test_utils", "utils", "test_model_fp16.xml")
        test_bin = os.path.join(base_path, "test_utils", "utils", "test_model_fp16.bin")
    else:
        test_xml = os.path.join(base_path, "test_utils", "utils", "test_model_fp32.xml")
        test_bin = os.path.join(base_path, "test_utils", "utils", "test_model_fp32.bin")
    return (test_xml, test_bin)


def model_onnx_path():
    base_path = os.path.dirname(__file__)
    test_onnx = os.path.join(base_path, "test_utils", "utils", "test_model.onnx")
    return test_onnx


def pytest_configure(config):

    # register additional markers
    config.addinivalue_line("markers", "skip_on_cpu: Skip test on CPU")
    config.addinivalue_line("markers", "skip_on_gpu: Skip test on GPU")
    config.addinivalue_line("markers", "skip_on_gna: Skip test on GNA")
    config.addinivalue_line("markers", "skip_on_hetero: Skip test on HETERO")
    config.addinivalue_line("markers", "skip_on_template: Skip test on TEMPLATE")
    config.addinivalue_line("markers", "onnx_coverage: Collect ONNX operator coverage")
    config.addinivalue_line("markers", "template_extension")
    config.addinivalue_line("markers", "dynamic_library: Runs tests only in dynamic libraries case")


@pytest.fixture(scope="session")
def device():
    return os.environ.get("TEST_DEVICE") if os.environ.get("TEST_DEVICE") else "CPU"
