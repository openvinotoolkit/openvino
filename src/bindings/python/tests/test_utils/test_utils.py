# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, List

import numpy as np
import openvino
import openvino.runtime.opset8 as ops
import pytest
from openvino.runtime import Model, Core, Shape, Type
from openvino.runtime.op import Parameter
from openvino.utils import deprecated


def get_test_model():
    element_type = Type.f32
    param = Parameter(element_type, Shape([1, 3, 22, 22]))
    relu = ops.relu(param)
    model = Model([relu], [param], "test")
    assert model is not None
    return model


def test_compare_models():
    try:
        from openvino.test_utils import compare_models
        model = get_test_model()
        status, _ = compare_models(model, model)
        assert status
    except RuntimeError:
        print("openvino.test_utils.compare_models is not available")  # noqa: T201


def generate_image(shape: Tuple = (1, 3, 32, 32), dtype: Union[str, np.dtype] = "float32") -> np.array:
    np.random.seed(42)
    return np.random.rand(*shape).astype(dtype)


def generate_relu_model(input_shape: List[int]) -> openvino.runtime.ie_api.CompiledModel:
    param = ops.parameter(input_shape, np.float32, name="parameter")
    relu = ops.relu(param, name="relu")
    model = Model([relu], [param], "test")
    model.get_ordered_ops()[2].friendly_name = "friendly"

    core = Core()
    return core.compile_model(model, "CPU", {})


def generate_add_model() -> openvino.pyopenvino.Model:
    param1 = ops.parameter(Shape([2, 1]), dtype=np.float32, name="data1")
    param2 = ops.parameter(Shape([2, 1]), dtype=np.float32, name="data2")
    add = ops.add(param1, param2)
    return Model(add, [param1, param2], "TestFunction")


def test_deprecation_decorator():
    @deprecated()
    def deprecated_function1(param1, param2=None):
        pass

    @deprecated(version="2025.4")
    def deprecated_function2(param1=None):
        pass

    @deprecated(message="Use another function instead")
    def deprecated_function3():
        pass

    @deprecated(version="2025.4", message="Use another function instead")
    def deprecated_function4():
        pass

    with pytest.warns(DeprecationWarning, match="deprecated_function1 is deprecated"):
        deprecated_function1("param1")
    with pytest.warns(DeprecationWarning, match="deprecated_function2 is deprecated and will be removed in version 2025.4"):
        deprecated_function2(param1=1)
    with pytest.warns(DeprecationWarning, match="deprecated_function3 is deprecated. Use another function instead"):
        deprecated_function3()
    with pytest.warns(DeprecationWarning, match="deprecated_function4 is deprecated and will be removed in version 2025.4. Use another function instead"):
        deprecated_function4()
