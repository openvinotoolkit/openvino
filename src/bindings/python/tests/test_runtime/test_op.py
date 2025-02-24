# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np

from openvino import (
    Model,
    op,
    opset13,
    Shape,
    Tensor,
    Type,
    serialize
)
from openvino.utils.postponed_constant import make_postponed_constant

class Maker:
    def __init__(self):
        self.called = False

    def __call__(self) -> Tensor:
        self.called = True
        tensor_data = np.asarray([2, 2, 2, 2], dtype=np.float32).astype(np.float32)
        return Tensor(tensor_data)

    def is_called(self):
        return self.called


def create_model(maker):
    input_shape = Shape([1, 1, 2, 2])
    param_node = op.Parameter(Type.f32, input_shape)

    postponned_constant = make_postponed_constant(Type.f32, input_shape, maker)

    add_1 = opset13.add(param_node, postponned_constant)

    const_2 = op.Constant(Type.f32, input_shape, [1, 2, 3, 4])
    add_2 = opset13.add(add_1, const_2)

    return Model(add_2, [param_node], 'test_model')

def test_postponned_constant():
    maker = Maker()
    model = create_model(maker)
    assert maker.is_called() == False
    serialize(model, 'out.xml', 'out.bin')
    assert maker.is_called() == True
    os.remove('out.xml')
    os.remove('out.bin')
