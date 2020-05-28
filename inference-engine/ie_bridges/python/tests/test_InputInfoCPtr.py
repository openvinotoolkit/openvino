import pytest

from openvino.inference_engine import InputInfoCPtr, DataPtr, IECore, TensorDesc
from conftest import model_path


test_net_xml, test_net_bin = model_path()


def test_name(device):
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.load_network(net, device, num_requests=5)
    assert isinstance(exec_net.input_info['data'], InputInfoCPtr)
    assert exec_net.input_info['data'].name == "data", "Incorrect name"
    del exec_net


def test_precision(device):
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.load_network(net, device, num_requests=5)
    assert isinstance(exec_net.input_info['data'], InputInfoCPtr)
    assert exec_net.input_info['data'].precision == "FP32", "Incorrect precision"
    del exec_net


def test_no_precision_setter(device):
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.load_network(net, device, num_requests=5)
    with pytest.raises(AttributeError) as e:
        exec_net.input_info['data'].precision = "I8"
    assert "attribute 'precision' of 'openvino.inference_engine.ie_api.InputInfoCPtr' " \
           "objects is not writable" in str(e.value)
    del exec_net


def test_input_data(device):
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.load_network(net, device, num_requests=5)
    assert isinstance(exec_net.input_info['data'], InputInfoCPtr)
    assert isinstance(exec_net.input_info['data'].input_data, DataPtr), "Incorrect precision for layer 'fc_out'"
    del exec_net


def test_tensor_desc(device):
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.load_network(net, device, num_requests=5)
    tensor_desc = exec_net.input_info['data'].tensor_desc
    assert isinstance(tensor_desc, TensorDesc)
    assert tensor_desc.layout == "NCHW"
