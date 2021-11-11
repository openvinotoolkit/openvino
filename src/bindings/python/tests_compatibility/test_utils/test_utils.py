# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.impl.op import Parameter
from openvino.impl import Function, Shape, Type
from openvino.opset8 import relu


def get_test_function():
    element_type = Type.f32
    param = Parameter(element_type, Shape([1, 3, 22, 22]))
    relu = relu(param)
    func = Function([relu], [param], 'test')
    assert func != None
    return func


def test_compare_networks():
    try:
        from openvino.test_utils import compare_functions
        func = get_test_function()
        status, msg = compare_functions(func, func)
        assert status
    except:
        print("openvino.test_utils.CompareNetworks is not available")
