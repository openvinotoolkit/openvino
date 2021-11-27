# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.runtime.descriptor import Tensor
from openvino.runtime.impl import Type, PartialShape


def test_tensor_descriptor_api():
    td = Tensor(Type.f32, PartialShape([1, 1, 1, 1]), "tensor_name")
    td.names = {"tensor_name"}
    assert "tensor_name" in td.names
    assert isinstance(td, Tensor)
    assert td.element_type == Type.f32
    assert td.partial_shape == PartialShape([1, 1, 1, 1])
    assert repr(td.shape) == "<Shape: {1, 1, 1, 1}>"
    assert td.size == 4
    assert td.any_name == "tensor_name"
