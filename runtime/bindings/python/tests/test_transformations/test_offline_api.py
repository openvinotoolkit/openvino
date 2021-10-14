# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.pyopenvino.offline_transformations import ApplyMOCTransformations, ApplyPOTTransformations, \
    ApplyLowLatencyTransformation, ApplyPruningTransformation, ApplyMakeStatefulTransformation

from ngraph.impl import Function, Shape
import ngraph as ng


def get_test_function():
    param = ng.parameter(Shape([1, 3, 22, 22]), name="parameter")
    relu = ng.relu(param)
    res = ng.result(relu, name="result")
    return Function([res], [param], "test")


def test_moc_transformations():
    function = get_test_function()

    ApplyMOCTransformations(function, False)

    assert function is not None
    assert len(function.get_ops()) == 3


def test_pot_transformations():
    function = get_test_function()

    ApplyPOTTransformations(function, "GNA")

    assert function is not None
    assert len(function.get_ops()) == 3


def test_low_latency_transformation():
    function = get_test_function()

    ApplyLowLatencyTransformation(function, True)

    assert function is not None
    assert len(function.get_ops()) == 3


def test_pruning_transformation():
    function = get_test_function()

    ApplyPruningTransformation(function)

    assert function is not None
    assert len(function.get_ops()) == 3


def test_make_stateful_transformations():
    function = get_test_function()

    ApplyMakeStatefulTransformation(function, {"parameter": "result"})

    assert function is not None
    assert len(function.get_parameters()) == 0
    assert len(function.get_results()) == 0
