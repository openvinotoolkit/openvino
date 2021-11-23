# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.impl import PartialShape, Type
from openvino.impl.op.util import VariableInfo, Variable


def test_info_as_property():
    info = VariableInfo()
    info.shape = PartialShape([1])
    info.type = Type.f32
    info.id = "test_id"
    variable = Variable(info)
    assert variable.info.shape == info.shape
    assert variable.info.type == info.type
    assert variable.info.id == info.id


def test_get_info():
    info = VariableInfo()
    info.shape = PartialShape([1])
    info.type = Type.f32
    info.id = "test_id"
    variable = Variable(info)
    assert variable.get_info().shape == info.shape
    assert variable.get_info().type == info.type
    assert variable.get_info().id == info.id


def test_info_update():
    info1 = VariableInfo()
    info1.shape = PartialShape([1])
    info1.type = Type.f32
    info1.id = "test_id"

    variable = Variable(info1)

    info2 = VariableInfo()
    info2.shape = PartialShape([2, 1])
    info2.type = Type.i64
    info2.id = "test_id2"

    variable.update(info2)
    assert variable.info.shape == info2.shape
    assert variable.info.type == info2.type
    assert variable.info.id == info2.id
