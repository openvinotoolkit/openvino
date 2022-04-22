# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.runtime import Model
from openvino.runtime import Shape, Type
from openvino.runtime.op import Parameter
from typing import Tuple, Union
import io
import openvino.runtime.opset8 as ops
import numpy as np
import onnx


def get_test_function():
    element_type = Type.f32
    param = Parameter(element_type, Shape([1, 3, 22, 22]))
    relu = ops.relu(param)
    func = Model([relu], [param], "test")
    assert func is not None
    return func


def test_compare_functions():
    try:
        from openvino.test_utils import compare_functions
        func = get_test_function()
        status, msg = compare_functions(func, func)
        assert status
    except RuntimeError:
        print("openvino.test_utils.compare_functions is not available")


def generate_image(shape: Tuple = (1, 3, 32, 32), dtype: Union[str, np.dtype] = "float32") -> np.array:
    np.random.seed(42)
    return np.random.rand(*shape).astype(dtype)
