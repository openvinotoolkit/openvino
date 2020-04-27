import pytest

from openvino.inference_engine import IECore, IENetLayer, DataPtr
from conftest import model_path


test_net_xml, test_net_bin = model_path()

def layer_out_data():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    return net.layers['19/Fused_Add_'].out_data[0]


def test_name():
    assert layer_out_data().name == "19/Fused_Add_", "Incorrect name for layer '19/Fused_Add_'"


def test_precision():
    assert layer_out_data().precision == "FP32", "Incorrect precision for layer '19/Fused_Add_'"


def test_precision_setter():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    net.layers['19/Fused_Add_'].out_data[0].precision = "I8"
    assert net.layers['19/Fused_Add_'].out_data[0].precision == "I8", "Incorrect precision for layer '19/Fused_Add_'"


def test_incorrect_precision_setter():
    with pytest.raises(ValueError) as e:
        layer_out_data().precision = "123"
    assert "Unsupported precision 123! List of supported precisions:" in str(e.value)


def test_layout():
    assert layer_out_data().layout == "NCHW", "Incorrect layout for layer '19/Fused_Add_'"


def test_initialized():
    assert layer_out_data().initialized, "Incorrect value for initialized property for layer '19/Fused_Add_'"
