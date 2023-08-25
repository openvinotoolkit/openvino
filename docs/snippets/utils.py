# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np

import openvino as ov
import openvino.runtime.opset12 as ops

import ngraph as ng
from ngraph.impl import Function


def get_model(input_shape = None, input_dtype=np.float32) -> ov.Model:
    if input_shape is None:
        input_shape = [1, 3, 32, 32]
    param = ops.parameter(input_shape, input_dtype, name="data")
    relu = ops.relu(param, name="relu")
    model = ov.Model([relu], [param], "test_model")

    assert model is not None
    return model


def get_ngraph_model(input_shape = None, input_dtype=np.float32):
    if input_shape is None:
        input_shape = [1, 3, 32, 32]
    param = ng.opset11.parameter(input_shape, input_dtype, name="data")
    relu = ng.opset11.relu(param, name="relu")
    model = Function([relu], [param], "test_model")

    assert model is not None
    return model


def get_image(shape = (1, 3, 32, 32), dtype = "float32") -> np.array:
    np.random.seed(42)
    return np.random.rand(*shape).astype(dtype)
