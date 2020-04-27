import pytest

from openvino.inference_engine import IETensorDesc


def test_init():
    tensor_desc = IETensorDesc("FP32", [1, 127, 127, 3], "NHWC")
    assert isinstance(tensor_desc, IETensorDesc)


def test_precision():
    tensor_desc = IETensorDesc("FP32", [1, 127, 127, 3], "NHWC")
    assert tensor_desc.precision == "FP32"


def test_layout():
    tensor_desc = IETensorDesc("FP32", [1, 127, 127, 3], "NHWC")
    assert tensor_desc.layout == "NHWC"


def test_dims():
    tensor_desc = IETensorDesc("FP32", [1, 127, 127, 3], "NHWC")
    assert tensor_desc.dims == [1, 127, 127, 3]


def test_incorrect_precision_setter():
    tensor_desc = IETensorDesc("FP32", [1, 127, 127, 3], "NHWC")
    with pytest.raises(ValueError) as e:
        tensor_desc.precision = "123"
    assert "Unsupported precision 123! List of supported precisions:" in str(e.value)


def test_incorrect_layout_setter():
    tensor_desc = IETensorDesc("FP32", [1, 127, 127, 3], "NHWC")
    with pytest.raises(ValueError) as e:
        tensor_desc.layout = "123"
    assert "Unsupported layout 123! List of supported layouts: " in str(e.value)


def test_init_incorrect_precision():
    with pytest.raises(ValueError) as e:
        IETensorDesc("123", [1, 127, 127, 3], "NHWC")
    assert "Unsupported precision 123! List of supported precisions: " in str(e.value)


def test_eq_operator():
    tensor_desc = IETensorDesc("FP32", [1, 3, 127, 127], "NHWC")
    tensor_desc_2 = IETensorDesc("FP32", [1, 3, 127, 127], "NHWC")
    assert tensor_desc == tensor_desc_2


def test_ne_operator():
    tensor_desc = IETensorDesc("FP32", [1, 3, 127, 127], "NHWC")
    tensor_desc_2 = IETensorDesc("FP32", [1, 3, 127, 127], "NCHW")
    assert tensor_desc != tensor_desc_2
