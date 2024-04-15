# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from openvino import PartialShape
from openvino.runtime import opset13 as ops
from openvino.runtime.passes import Matcher, WrapType, Or, AnyInput, Optional
from openvino.runtime.passes import (
    consumers_count,
    has_static_dim,
    has_static_dims,
    has_static_shape,
    has_static_rank,
    type_matches,
    type_matches_any,
)
from openvino.runtime.utils.types import get_element_type

from tests.test_transformations.utils.utils import expect_exception


def test_wrap_type_pattern_type():
    last_opset_number = 15
    for i in range(1, last_opset_number + 1):
        WrapType(f"opset{i}.Parameter")
        WrapType(f"opset{i}::Parameter")

    # Negative check not to forget to update opset map in get_type function
    expect_exception(lambda: WrapType(f"opset{last_opset_number + 1}.Parameter"),
                     f"Unsupported opset type: opset{last_opset_number + 1}")

    # Generic negative test cases
    expect_exception(lambda: WrapType(""))
    expect_exception(lambda: WrapType("ops"))
    expect_exception(lambda: WrapType("Parameter"))
    expect_exception(lambda: WrapType("opset.Parameter"))
    expect_exception(lambda: WrapType("ops,Parameter"))
    expect_exception(lambda: WrapType("Parameter.ops"))


def test_wrap_type_ctors():
    param = ops.parameter(PartialShape([1, 3, 22, 22]))
    relu = ops.relu(param.output(0))
    slope = ops.parameter(PartialShape([]))
    prelu = ops.prelu(param.output(0), slope.output(0))

    matcher = Matcher(WrapType(["opset13.Relu", "opset13.PRelu"]), "FindActivation")
    assert matcher.match(relu)
    assert matcher.match(prelu)

    matcher = Matcher(WrapType(["opset13.Relu", "opset13.PRelu"],
                      WrapType("opset13.Parameter").output(0)), "FindActivation")
    assert matcher.match(relu)


def test_or():
    param = ops.parameter(PartialShape([1, 3, 22, 22]))
    relu = ops.relu(param.output(0))
    slope = ops.parameter(PartialShape([]))
    prelu = ops.prelu(param.output(0), slope.output(0))

    matcher = Matcher(Or([WrapType("opset13.Relu"), WrapType("opset13.PRelu")]), "FindActivation")
    assert matcher.match(relu)
    assert matcher.match(prelu)


def test_any_input():
    param = ops.parameter(PartialShape([1, 3, 22, 22]))
    relu = ops.relu(param.output(0))
    slope = ops.parameter(PartialShape([]))
    prelu = ops.prelu(param.output(0), slope.output(0))

    matcher = Matcher(WrapType("opset13.PRelu", [AnyInput(), AnyInput()]), "FindActivation")
    assert not matcher.match(relu)
    assert matcher.match(prelu)


def test_any_input_predicate():
    param = ops.parameter(PartialShape([1, 3, 22, 22]))
    slope = ops.parameter(PartialShape([]))

    matcher = Matcher(AnyInput(lambda output: len(output.get_shape()) == 4), "FindActivation")
    assert matcher.match(param)
    assert not matcher.match(slope)


def test_optional_full_match():
    model_input = ops.parameter(PartialShape.dynamic())
    model_abs = ops.abs(model_input)
    model_relu = ops.relu(model_abs.output(0))

    pattern_abs = Optional(["opset13.Abs"])
    pattern_relu = ops.relu(pattern_abs.output(0))

    matcher = Matcher(pattern_relu, "FindRelu")
    assert matcher.match(model_relu)


@pytest.mark.skip("Optional is not working properly yet CVS-136454")
def test_optional_half_match():
    model_input = ops.parameter(PartialShape.dynamic())
    model_relu = ops.relu(model_input)
    model_relu1 = ops.relu(model_relu.output(0))

    pattern_abs = Optional(["opset13.Abs"])
    pattern_relu = ops.relu(pattern_abs.output(0))

    matcher = Matcher(pattern_relu, "FindRelu")
    assert matcher.match(model_relu1)


@pytest.mark.skip("Optional is not working properly yet CVS-136454")
def test_optional_one_node():
    model_input = ops.parameter(PartialShape.dynamic())
    model_relu = ops.relu(model_input)
    model_abs = ops.abs(model_input)

    assert Matcher(Optional(["opset13.Relu"]), "OneNodeTest").match(model_relu)
    assert not Matcher(Optional(["opset13.Abs"]), "OneNodeTest").match(model_relu)

    assert not Matcher(Optional(["opset13.Relu"]), "OneNodeTest").match(model_abs)

    assert Matcher(Optional(["opset13.Parameter"]), "OneNodeTest").match(ops.parameter(PartialShape.dynamic()))
    assert not Matcher(Optional(["opset13.Relu"]), "OneNodeTest").match(ops.parameter(PartialShape.dynamic()))


@pytest.mark.skip("Optional is not working properly yet CVS-136454")
def test_optional_predicate():
    model_input = ops.parameter(PartialShape.dynamic())
    model_add = ops.add(model_input, model_input)
    model_relu = ops.relu(model_add.output(0))
    model_abs = ops.abs(model_add.output(0))

    assert Matcher(Optional(["opset13.Relu"], lambda x: True), "TestInputPredicate").match(model_relu)
    assert not Matcher(Optional(["opset13.Relu"], lambda x: False), "TestInputPredicate").match(model_relu)
    assert Matcher(Optional(["opset13.Add"], consumers_count(2)), "FindPredicate").match(model_add)
    assert not Matcher(Optional(["opset13.Add"], consumers_count(1)), "FindPredicate").match(model_add)
    assert Matcher(Optional(["opset13.Abs", "opset13.Result"], consumers_count(0)), "FindPredicate").match(model_abs)


def test_optional_with_input():
    model_input = ops.parameter(PartialShape.dynamic())
    model_add = ops.add(model_input, model_input)
    model_relu = ops.relu(model_add.output(0))

    assert Matcher(Optional(["opset13.Relu"], model_add.output(0)), "TestInput").match(model_relu)
    assert not Matcher(Optional(["opset13.Cos"], model_add.output(0)), "TestInput").match(model_relu)


def test_optional_with_input_and_predicate():
    model_input = ops.parameter(PartialShape.dynamic())
    model_add = ops.add(model_input, model_input)
    model_relu = ops.relu(model_add.output(0))

    pattern_add = ops.add(AnyInput(), AnyInput())

    assert Matcher(Optional(["opset13.Relu"], pattern_add.output(0), lambda x: True), "TestInputPredicate").match(model_relu)
    assert not Matcher(Optional(["opset13.Relu"], pattern_add.output(0), lambda x: False), "TestInputPredicate").match(model_relu)


def test_optional_with_input_node():
    model_input = ops.parameter(PartialShape.dynamic())
    model_add = ops.add(model_input, model_input)
    model_relu = ops.relu(model_add.output(0))

    assert Matcher(Optional(["opset13.Relu"], model_add), "TestInputNode").match(model_relu)
    assert not Matcher(Optional(["opset13.Cos"], model_add), "TestInputNode").match(model_relu)


def test_optional_with_input_node_and_predicate():
    model_input = ops.parameter(PartialShape.dynamic())
    model_add = ops.add(model_input, model_input)
    model_relu = ops.relu(model_add.output(0))

    assert Matcher(Optional(["opset13.Relu"], model_add, lambda x: True), "TestInputNodePredicate").match(model_relu)
    assert not Matcher(Optional(["opset13.Relu"], model_add, lambda x: False), "TestInputNodePredicate").match(model_relu)
    assert not Matcher(Optional(["opset13.Cos"], model_add, lambda x: True), "TestInputNodePredicate").match(model_relu)


def test_all_predicates():
    static_param = ops.parameter(PartialShape([1, 3, 22, 22]), np.float32)
    dynamic_param = ops.parameter(PartialShape([-1, 6]), np.compat.long)
    fully_dynamic_param = ops.parameter(PartialShape.dynamic())

    assert Matcher(WrapType("opset13.Parameter", consumers_count(0)), "Test").match(static_param)
    assert not Matcher(WrapType("opset13.Parameter", consumers_count(1)), "Test").match(static_param)

    assert Matcher(WrapType("opset13.Parameter", has_static_dim(1)), "Test").match(static_param)
    assert not Matcher(WrapType("opset13.Parameter", has_static_dim(0)), "Test").match(dynamic_param)

    assert Matcher(WrapType("opset13.Parameter", has_static_dims([0, 3])), "Test").match(static_param)
    assert not Matcher(WrapType("opset13.Parameter", has_static_dims([0, 1])), "Test").match(dynamic_param)

    assert Matcher(WrapType("opset13.Parameter", has_static_shape()), "Test").match(static_param)
    assert not Matcher(WrapType("opset13.Parameter", has_static_shape()), "Test").match(dynamic_param)

    assert Matcher(WrapType("opset13.Parameter", has_static_rank()), "Test").match(dynamic_param)
    assert not Matcher(WrapType("opset13.Parameter", has_static_rank()), "Test").match(fully_dynamic_param)

    assert Matcher(WrapType("opset13.Parameter",
                            type_matches(get_element_type(np.float32))), "Test").match(static_param)
    assert not Matcher(WrapType("opset13.Parameter",
                                type_matches(get_element_type(np.float32))), "Test").match(dynamic_param)

    assert Matcher(WrapType("opset13.Parameter",  # noqa: ECE001
                            type_matches_any([get_element_type(np.float32),
                                              get_element_type(np.compat.long)])), "Test").match(static_param)
    assert Matcher(WrapType("opset13.Parameter",  # noqa: ECE001
                            type_matches_any([get_element_type(np.float32),
                                              get_element_type(np.compat.long)])), "Test").match(dynamic_param)
