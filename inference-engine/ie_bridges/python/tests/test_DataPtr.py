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
