# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino import Model, PartialShape
from openvino import opset13 as ops
from openvino.utils import replace_node, replace_output_update_name


def get_relu_model():
    # Parameter->Relu->Result
    param = ops.parameter(PartialShape([1, 3, 22, 22]), name="parameter")
    relu = ops.relu(param.output(0))
    res = ops.result(relu.output(0), name="result")
    return Model([res], [param], "test")


def test_output_replace():
    param = ops.parameter(PartialShape([1, 3, 22, 22]), name="parameter")
    relu = ops.relu(param.output(0))
    res = ops.result(relu.output(0), name="result")

    exp = ops.exp(param.output(0))
    relu.output(0).replace(exp.output(0))

    assert res.input_value(0).get_node() == exp


def test_replace_source_output():
    param = ops.parameter(PartialShape([1, 3, 22, 22]), name="parameter")
    relu = ops.relu(param.output(0))
    res = ops.result(relu.output(0), name="result")

    exp = ops.exp(param.output(0))
    res.input(0).replace_source_output(exp.output(0))

    assert len(exp.output(0).get_target_inputs()) == 1
    assert len(relu.output(0).get_target_inputs()) == 0
    target_inputs = exp.output(0).get_target_inputs()
    assert next(iter(target_inputs)).get_node() == res


def test_replace_node():
    param = ops.parameter(PartialShape([1, 3, 22, 22]), name="parameter")
    relu = ops.relu(param.output(0))
    res = ops.result(relu.output(0), name="result")

    exp = ops.exp(param.output(0))
    replace_node(relu, exp)

    assert res.input_value(0).get_node() == exp


def test_replace_output_update_name():
    param = ops.parameter(PartialShape([1, 3, 22, 22]), name="parameter")
    relu = ops.relu(param.output(0))
    exp = ops.exp(relu.output(0))
    res = ops.result(exp.output(0), name="result")

    replace_output_update_name(exp.output(0), exp.input_value(0))

    assert res.input_value(0).get_node() == exp
