# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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


def test_eq_operator():
    tensor_desc = TensorDesc("FP32", [1, 3, 127, 127], "NHWC")
    tensor_desc_2 = TensorDesc("FP32", [1, 3, 127, 127], "NHWC")
    assert tensor_desc == tensor_desc_2


def test_ne_operator():
    tensor_desc = TensorDesc("FP32", [1, 3, 127, 127], "NHWC")
    tensor_desc_2 = TensorDesc("FP32", [1, 3, 127, 127], "NCHW")
    assert tensor_desc != tensor_desc_2
