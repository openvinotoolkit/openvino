# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.runtime import Model, PartialShape, opset8
from openvino.runtime.utils import replace_node, replace_output_update_name


def get_test_function():
    # Parameter->Relu->Result
    param = opset8.parameter(PartialShape([1, 3, 22, 22]), name="parameter")
    relu = opset8.relu(param.output(0))
    res = opset8.result(relu.output(0), name="result")
    return Model([res], [param], "test")


def test_output_replace():
    param = opset8.parameter(PartialShape([1, 3, 22, 22]), name="parameter")
    relu = opset8.relu(param.output(0))
    res = opset8.result(relu.output(0), name="result")

    exp = opset8.exp(param.output(0))
    relu.output(0).replace(exp.output(0))

    assert res.input_value(0).get_node() == exp


def test_replace_source_output():
    param = opset8.parameter(PartialShape([1, 3, 22, 22]), name="parameter")
    relu = opset8.relu(param.output(0))
    res = opset8.result(relu.output(0), name="result")

    exp = opset8.exp(param.output(0))
    res.input(0).replace_source_output(exp.output(0))

    assert len(exp.output(0).get_target_inputs()) == 1
    assert len(relu.output(0).get_target_inputs()) == 0
    assert next(iter(exp.output(0).get_target_inputs())).get_node() == res


def test_replace_node():
    param = opset8.parameter(PartialShape([1, 3, 22, 22]), name="parameter")
    relu = opset8.relu(param.output(0))
    res = opset8.result(relu.output(0), name="result")

    exp = opset8.exp(param.output(0))
    replace_node(relu, exp)

    assert res.input_value(0).get_node() == exp


def test_replace_output_update_name():
    param = opset8.parameter(PartialShape([1, 3, 22, 22]), name="parameter")
    relu = opset8.relu(param.output(0))
    exp = opset8.exp(relu.output(0))
    res = opset8.result(exp.output(0), name="result")

    replace_output_update_name(exp.output(0), exp.input_value(0))

    assert res.input_value(0).get_node() == exp
