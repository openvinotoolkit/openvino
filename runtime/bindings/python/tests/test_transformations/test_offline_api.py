# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.pyopenvino.offline_transformations import ApplyMOCTransformations, ApplyPOTTransformations, \
    ApplyLowLatencyTransformation, ApplyPruningTransformation

from openvino.impl.op import Parameter
from openvino.impl import Function, Shape, Type
from openvino.opset8 import relu


def get_test_function():
    element_type = Type.f32
    param = Parameter(element_type, Shape([1, 3, 22, 22]))
    op = relu(param)
    return Function([op], [param], 'test')


def test_moc_transformations():
    f = get_test_function()

    ApplyMOCTransformations(f, False)

    assert f != None
    assert len(f.get_ops()) == 3


def test_pot_transformations():
    f = get_test_function()

    ApplyPOTTransformations(f, "GNA")

    assert f != None
    assert len(f.get_ops()) == 3


def test_low_latency_transformation():
    f = get_test_function()

    ApplyLowLatencyTransformation(f, True)

    assert f != None
    assert len(f.get_ops()) == 3


def test_pruning_transformation():
    f = get_test_function()

    ApplyPruningTransformation(f)

    assert f != None
    assert len(f.get_ops()) == 3
