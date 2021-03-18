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

from openvino.inference_engine import TensorDesc


def test_init():
    tensor_desc = TensorDesc("FP32", [1, 127, 127, 3], "NHWC")
    assert isinstance(tensor_desc, TensorDesc)


def test_precision():
    tensor_desc = TensorDesc("FP32", [1, 127, 127, 3], "NHWC")
    assert tensor_desc.precision == "FP32"


def test_layout():
    tensor_desc = TensorDesc("FP32", [1, 127, 127, 3], "NHWC")
    assert tensor_desc.layout == "NHWC"


def test_dims():
    tensor_desc = TensorDesc("FP32", [1, 127, 127, 3], "NHWC")
    assert tensor_desc.dims == [1, 127, 127, 3]


def test_incorrect_precision_setter():
    tensor_desc = TensorDesc("FP32", [1, 127, 127, 3], "NHWC")
    with pytest.raises(ValueError) as e:
        tensor_desc.precision = "123"
    assert "Unsupported precision 123! List of supported precisions:" in str(e.value)


def test_incorrect_layout_setter():
    tensor_desc = TensorDesc("FP32", [1, 127, 127, 3], "NHWC")
    with pytest.raises(ValueError) as e:
        tensor_desc.layout = "123"
    assert "Unsupported layout 123! List of supported layouts: " in str(e.value)


def test_init_incorrect_precision():
    with pytest.raises(ValueError) as e:
        TensorDesc("123", [1, 127, 127, 3], "NHWC")
    assert "Unsupported precision 123! List of supported precisions: " in str(e.value)


def test_eq_operator():
    tensor_desc = TensorDesc("FP32", [1, 3, 127, 127], "NHWC")
    tensor_desc_2 = TensorDesc("FP32", [1, 3, 127, 127], "NHWC")
    assert tensor_desc == tensor_desc_2


def test_ne_operator():
    tensor_desc = TensorDesc("FP32", [1, 3, 127, 127], "NHWC")
    tensor_desc_2 = TensorDesc("FP32", [1, 3, 127, 127], "NCHW")
    assert tensor_desc != tensor_desc_2
