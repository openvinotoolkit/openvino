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

from openvino.inference_engine import IECore, DataPtr
from conftest import model_path


test_net_xml, test_net_bin = model_path()


def layer_out_data():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    return net.outputs['fc_out']


def test_name():
    assert layer_out_data().name == 'fc_out', "Incorrect name for layer 'fc_out'"


def test_precision():
    assert layer_out_data().precision == "FP32", "Incorrect precision for layer 'fc_out'"


def test_precision_setter():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    net.outputs['fc_out'].precision = "I8"
    assert net.outputs['fc_out'].precision == "I8", "Incorrect precision for layer 'fc_out'"


def test_incorrect_precision_setter():
    with pytest.raises(ValueError) as e:
        layer_out_data().precision = "123"
    assert "Unsupported precision 123! List of supported precisions:" in str(e.value)


def test_layout():
    assert layer_out_data().layout == "NC", "Incorrect layout for layer 'fc_out'"


def test_initialized():
    assert layer_out_data().initialized, "Incorrect value for initialized property for layer 'fc_out'"
