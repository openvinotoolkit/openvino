# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino
from openvino.runtime import Model, Core, Shape, Type
from openvino.runtime.op import Parameter
from typing import Tuple, Union, List
import openvino.runtime.opset8 as ops
import numpy as np


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
        status, _ = compare_functions(func, func)
        assert status
    except RuntimeError:
        print("openvino.test_utils.compare_functions is not available")


def generate_image(shape: Tuple = (1, 3, 32, 32), dtype: Union[str, np.dtype] = "float32") -> np.array:
    np.random.seed(42)
    return np.random.rand(*shape).astype(dtype)


def generate_model(input_shape: List[int]) -> openvino.runtime.ie_api.CompiledModel:
    param = ops.parameter(input_shape, np.float32, name="parameter")
    relu = ops.relu(param, name="relu")
    func = Model([relu], [param], "test")
    func.get_ordered_ops()[2].friendly_name = "friendly"

    core = Core()
    return core.compile_model(func, "CPU", {})
