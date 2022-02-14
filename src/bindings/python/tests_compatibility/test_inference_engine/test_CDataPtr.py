# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from openvino.inference_engine import CDataPtr, IECore
from tests_compatibility.conftest import model_path, create_relu
import ngraph as ng


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


@pytest.mark.skip(reason="Old Python API seg faults during dynamic shape inference")
def test_is_dynamic():
    function = create_relu([-1, 3, 20, 20])
    net = ng.function_to_cnn(function)
    ie = IECore()
    ie.register_plugin("openvino_template_plugin", "TEMPLATE")
    exec_net = ie.load_network(net, "TEMPLATE")
    assert exec_net.outputs["out"].is_dynamic
    p_shape = ng.partial_shape_from_data(exec_net.outputs["out"])
    assert isinstance(p_shape, ng.impl.PartialShape)
    with pytest.raises(RuntimeError) as e:
       exec_net.outputs["out"].shape
    assert  "Cannot return dims for Data with dynamic shapes!" in str(e.value)
