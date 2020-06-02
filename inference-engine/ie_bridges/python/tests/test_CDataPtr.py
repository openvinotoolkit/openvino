import os
import pytest

from openvino.inference_engine import CDataPtr, IECore

SAMPLENET_XML = os.path.join(os.path.dirname(__file__), 'test_data', 'models', 'test_model.xml')
SAMPLENET_BIN = os.path.join(os.path.dirname(__file__), 'test_data', 'models', 'test_model.bin')


def test_name(device):
    ie = IECore()
    net = ie.read_network(model=SAMPLENET_XML, weights=SAMPLENET_BIN)
    exec_net = ie.load_network(net, device, num_requests=5)
    assert isinstance(exec_net.outputs['fc_out'], CDataPtr)
    assert exec_net.outputs['fc_out'].name == "fc_out", "Incorrect name for layer 'fc_out'"


def test_precision(device):
    ie = IECore()
    net = ie.read_network(model=SAMPLENET_XML, weights=SAMPLENET_BIN)
    exec_net = ie.load_network(net, device, num_requests=5)
    assert isinstance(exec_net.outputs['fc_out'], CDataPtr)
    assert exec_net.outputs['fc_out'].precision == "FP32", "Incorrect precision for layer 'fc_out'"


def test_no_precision_setter(device):
    ie = IECore()
    net = ie.read_network(model=SAMPLENET_XML, weights=SAMPLENET_BIN)
    exec_net = ie.load_network(net, device, num_requests=5)
    with pytest.raises(AttributeError) as e:
        exec_net.outputs['fc_out'].precision = "I8"
    assert "attribute 'precision' of 'openvino.inference_engine.ie_api.CDataPtr' objects is not writable" in str(e.value)


def test_layout(device):
    ie = IECore()
    net = ie.read_network(model=SAMPLENET_XML, weights=SAMPLENET_BIN)
    exec_net = ie.load_network(net, device, num_requests=5)
    assert exec_net.outputs['fc_out'].layout == "NC", "Incorrect layout for layer 'fc_out"


def test_no_layout_setter(device):
    ie = IECore()
    net = ie.read_network(model=SAMPLENET_XML, weights=SAMPLENET_BIN)
    exec_net = ie.load_network(net, device, num_requests=5)
    with pytest.raises(AttributeError) as e:
        exec_net.outputs['fc_out'].layout = "CN"
    assert "attribute 'layout' of 'openvino.inference_engine.ie_api.CDataPtr' objects is not writable" in str(e.value)


def test_initialized(device):
    ie = IECore()
    net = ie.read_network(model=SAMPLENET_XML, weights=SAMPLENET_BIN)
    exec_net = ie.load_network(net, device, num_requests=5)
    assert exec_net.outputs['fc_out'].initialized, "Incorrect value for initialized property for layer 'fc_out"
