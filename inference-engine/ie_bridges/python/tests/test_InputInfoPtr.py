"""
 Copyright (C) 2018-2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import pytest

from openvino.inference_engine import InputInfoPtr, PreProcessInfo, DataPtr, IECore, TensorDesc, ColorFormat
from conftest import model_path


test_net_xml, test_net_bin = model_path()


def get_input_info():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    return net.input_info["data"]


def test_input_info():
    assert isinstance(get_input_info(), InputInfoPtr)


def test_input_data():
    assert isinstance(get_input_info().input_data, DataPtr)


def test_input_data_setter():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    input_info = net.input_info["data"]
    other_input_data = net.outputs["fc_out"]
    input_info.input_data = other_input_data
    assert input_info.input_data.name == "fc_out"


def test_incorrect_input_info_setter():
    with pytest.raises(TypeError) as e:
        get_input_info().input_data = "dfds"
    assert "Argument 'input_ptr' has incorrect type" in str(e.value)


def test_name():
    assert get_input_info().name == "data"


def test_precision():
    assert get_input_info().precision == "FP32"


def test_precision_setter():
    input_info = get_input_info()
    input_info.precision = "I8"
    assert input_info.precision == "I8", "Incorrect precision"


def test_incorrect_precision_setter():
    with pytest.raises(ValueError) as e:
        get_input_info().precision = "123"
    assert "Unsupported precision 123! List of supported precisions:" in str(e.value)


def test_layout():
    assert get_input_info().layout == "NCHW"


def test_layout_setter():
    input_info = get_input_info()
    input_info.layout = "NHWC"
    assert input_info.layout == "NHWC", "Incorrect layout"


def test_incorrect_layout_setter():
    with pytest.raises(ValueError) as e:
        get_input_info().layout = "123"
    assert "Unsupported layout 123! List of supported layouts:" in str(e.value)


def test_preprocess_info():
    preprocess_info = get_input_info().preprocess_info
    assert isinstance(preprocess_info, PreProcessInfo)
    assert preprocess_info.color_format == ColorFormat.RAW


def test_tensor_desc():
    tensor_desc = get_input_info().tensor_desc
    assert isinstance(tensor_desc, TensorDesc)
    assert tensor_desc.layout == "NCHW"
