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

from openvino.inference_engine import CDataPtr, IECore
from conftest import model_path


test_net_xml, test_net_bin = model_path()


def test_name(device):
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.load_network(net, device, num_requests=5)
    assert isinstance(exec_net.outputs['fc_out'], CDataPtr)
    assert exec_net.outputs['fc_out'].name == "fc_out", "Incorrect name for layer 'fc_out'"


def test_precision(device):
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.load_network(net, device, num_requests=5)
    assert isinstance(exec_net.outputs['fc_out'], CDataPtr)
    assert exec_net.outputs['fc_out'].precision == "FP32", "Incorrect precision for layer 'fc_out'"


def test_no_precision_setter(device):
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.load_network(net, device, num_requests=5)
    with pytest.raises(AttributeError) as e:
        exec_net.outputs['fc_out'].precision = "I8"
    assert "attribute 'precision' of 'openvino.inference_engine.ie_api.CDataPtr' objects is not writable" in str(e.value)


def test_layout(device):
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.load_network(net, device, num_requests=5)
    assert exec_net.outputs['fc_out'].layout == "NC", "Incorrect layout for layer 'fc_out"


def test_no_layout_setter(device):
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.load_network(net, device, num_requests=5)
    with pytest.raises(AttributeError) as e:
        exec_net.outputs['fc_out'].layout = "CN"
    assert "attribute 'layout' of 'openvino.inference_engine.ie_api.CDataPtr' objects is not writable" in str(e.value)


def test_initialized(device):
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.load_network(net, device, num_requests=5)
    assert exec_net.outputs['fc_out'].initialized, "Incorrect value for initialized property for layer 'fc_out"
