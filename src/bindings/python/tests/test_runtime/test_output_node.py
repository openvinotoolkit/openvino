# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import openvino.opset13 as ops
from openvino import Type, Tensor, Symbol
import numpy as np


def test_output_replace(device):
    param = ops.parameter([1, 64], Type.i64)
    param.output(0).get_tensor().set_names({"a", "b"})
    relu = ops.relu(param)
    relu.output(0).get_tensor().set_names({"c", "d"})

    new_relu = ops.relu(param)
    new_relu.output(0).get_tensor().set_names({"f"})

    relu.output(0).replace(new_relu.output(0))

    assert new_relu.output(0).get_tensor().get_names() == {"c", "d", "f"}


def test_output_names():
    param = ops.parameter([1, 64], Type.i64)

    names = {"param1", "data1"}
    param.output(0).set_names(names)
    assert param.output(0).get_names() == names

    more_names = {"yet_another_name", "input1"}
    param.output(0).add_names(more_names)
    assert param.output(0).get_names() == names.union(more_names)


def test_tensor_bounds():
    param = ops.parameter([1, 64], Type.f32)
    tensor = param.output(0).get_tensor()
    values = Tensor(np.zeros([1, 64], dtype=np.float32))
    tensor.set_lower_value(values)
    tensor.set_upper_value(values)
    assert np.array_equal(tensor.get_lower_value().data, values.data)
    assert np.array_equal(tensor.get_upper_value().data, values.data)


def test_output_symbol():
    param = ops.parameter([1, 64], Type.f32)
    tensor = param.output(0).get_tensor()
    values = [Symbol() for i in range(64)]
    tensor.set_value_symbol(values)
    gotten_values = tensor.get_value_symbol()
    assert gotten_values == values
