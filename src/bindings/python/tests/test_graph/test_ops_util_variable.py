# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino import PartialShape, Type
from openvino.op.util import VariableInfo, Variable


def test_info_as_property():
    info = VariableInfo()
    info.data_shape = PartialShape([1])
    info.data_type = Type.f32
    info.variable_id = "test_id"
    var_from_info = Variable(info)
    assert var_from_info.info.data_shape == info.data_shape
    assert var_from_info.info.data_type == info.data_type
    assert var_from_info.info.variable_id == info.variable_id


def test_get_info():
    info = VariableInfo()
    info.data_shape = PartialShape([1])
    info.data_type = Type.f32
    info.variable_id = "test_id"
    var_from_info = Variable(info)
    assert var_from_info.get_info().data_shape == info.data_shape
    assert var_from_info.get_info().data_type == info.data_type
    assert var_from_info.get_info().variable_id == info.variable_id


def test_info_update():
    info1 = VariableInfo()
    info1.data_shape = PartialShape([1])
    info1.data_type = Type.f32
    info1.variable_id = "test_id"

    var_from_info = Variable(info1)

    info2 = VariableInfo()
    info2.data_shape = PartialShape([2, 1])
    info2.data_type = Type.i64
    info2.variable_id = "test_id2"

    var_from_info.update(info2)
    assert var_from_info.info.data_shape == info2.data_shape
    assert var_from_info.info.data_type == info2.data_type
    assert var_from_info.info.variable_id == info2.variable_id
