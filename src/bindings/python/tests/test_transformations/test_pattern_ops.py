# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from openvino import PartialShape, Symbol, Dimension
from openvino import opset13 as ops
from openvino.passes import Matcher, WrapType, Or, AnyInput, Optional
from openvino.passes import (
    consumers_count,
    has_static_dim,
    has_static_dims,
    has_static_shape,
    has_static_rank,
    type_matches,
    type_matches_any,
    shape_matches,
)
from openvino.utils.types import get_element_type

from tests.test_transformations.utils.utils import expect_exception


def test_simple_model_and_pattern():
    # Create a sample model
    model_param1 = ops.parameter(PartialShape([2, 2]))
    model_param2 = ops.parameter(PartialShape([2, 2]))
    model_add = ops.add(model_param1, model_param2)
    model_param3 = ops.parameter(PartialShape([2, 2]))
    model_mul = ops.matmul(model_add, model_param3, False, False)
    model_abs = ops.abs(model_mul)
    model_relu = ops.relu(model_abs)
    model_result = ops.result(model_relu) # noqa

    # Create a sample pattern
    pattern_mul = ops.matmul(AnyInput(), AnyInput(), False, False)
    pattern_abs = ops.abs(pattern_mul)
    pattern_relu = ops.relu(pattern_abs)

    # Create a matcher and try to match the nodes
    matcher = Matcher(pattern_relu, "FindPattern")

    # Should perfectly match
    assert matcher.match(model_relu)


def test_simple_model_and_pattern_wrap_type():
    model_param1 = ops.parameter(PartialShape([2, 2]))
    model_param2 = ops.parameter(PartialShape([2, 2]))
    model_add = ops.add(model_param1, model_param2)
    model_param3 = ops.parameter(PartialShape([2, 2]))
    model_mul = ops.matmul(model_add, model_param3, False, False)
    model_abs = ops.abs(model_mul)
    model_relu = ops.relu(model_abs)
    model_result = ops.result(model_relu) # noqa

    # Create a sample pattern
    pattern_mul = WrapType("opset13.MatMul", [AnyInput(), AnyInput()])
    pattern_abs = WrapType("opset13.Abs", pattern_mul)
    pattern_relu = WrapType("opset13.Relu", pattern_abs)

    # Create a matcher and try to match the nodes
    matcher = Matcher(pattern_relu, "FindPattern")

    # Should perfectly match
    assert matcher.match(model_relu)


def test_wrap_type_list():
    model_param1 = ops.parameter(PartialShape([2, 2]))
    model_param2 = ops.parameter(PartialShape([2, 2]))
    model_add = ops.add(model_param1, model_param2)
    model_param3 = ops.parameter(PartialShape([2, 2]))
    model_mul = ops.matmul(model_add, model_param3, False, False)
    model_abs = ops.abs(model_mul)
    model_relu = ops.relu(model_abs)
    model_result = ops.result(model_relu) # noqa
    model_sig = ops.sigmoid(model_abs)  # Note that we've added a Sigmoid node after Abs
    model_result1 = ops.result(model_sig) # noqa

    # Create a sample pattern
    pattern_mul = WrapType("opset13.MatMul", [AnyInput(), AnyInput()])
    pattern_abs = WrapType("opset13.Abs", pattern_mul)
    pattern_relu = WrapType(["opset13.Relu", "opset13.Sigmoid"], pattern_abs)

    # Create a matcher and try to match the nodes
    matcher = Matcher(pattern_relu, "FindPattern")

    # The same pattern perfectly matches 2 different nodes
    assert matcher.match(model_relu)
    assert matcher.match(model_sig)


def test_pattern_or():
    model_param1 = ops.parameter(PartialShape([2, 2]))
    model_param2 = ops.parameter(PartialShape([2, 2]))
    model_add = ops.add(model_param1, model_param2)
    model_param3 = ops.parameter(PartialShape([2, 2]))
    model_mul = ops.matmul(model_add, model_param3, False, False)
    model_abs = ops.abs(model_mul)
    model_relu = ops.relu(model_abs)
    model_result = ops.result(model_relu) # noqa

    # Create a red branch
    red_pattern_add = WrapType("opset13.Add", [AnyInput(), AnyInput()])
    red_pattern_relu = WrapType("opset13.Relu", red_pattern_add)
    red_pattern_sigmoid = WrapType(["opset13.Sigmoid"], red_pattern_relu)

    # Create a blue branch
    blue_pattern_mul = WrapType("opset13.MatMul", [AnyInput(), AnyInput()])
    blue_pattern_abs = WrapType("opset13.Abs", blue_pattern_mul)
    blue_pattern_relu = WrapType(["opset13.Relu"], blue_pattern_abs)

    # Create Or node
    pattern_or = Or([red_pattern_sigmoid, blue_pattern_relu])

    # Create a matcher and try to match the nodes
    matcher = Matcher(pattern_or, "FindPattern")

    # The same pattern perfectly matches 2 different nodes
    assert matcher.match(model_relu)


def test_pattern_optional_middle():
    model_param1 = ops.parameter(PartialShape([2, 2]))
    model_param2 = ops.parameter(PartialShape([2, 2]))
    model_add = ops.add(model_param1, model_param2)
    model_param3 = ops.parameter(PartialShape([2, 2]))
    model_mul = ops.matmul(model_add, model_param3, False, False)
    model_abs = ops.abs(model_mul)
    model_relu = ops.relu(model_abs)
    model_result = ops.result(model_relu) # noqa

    # Create a sample pattern with an Optional node in the middle
    pattern_mul = WrapType("opset13.MatMul", [AnyInput(), AnyInput()])
    pattern_abs = WrapType("opset13.Abs", pattern_mul)
    pattern_sig_opt = Optional(["opset13.Sigmoid"], pattern_abs)
    pattern_relu = WrapType("opset13.Relu", pattern_sig_opt)

    # Create a matcher and try to match the nodes
    matcher = Matcher(pattern_relu, "FindPattern")

    # Should perfectly match
    assert matcher.match(model_relu)


def test_pattern_optional_top():
    model_param1 = ops.parameter(PartialShape([2, 2]))
    model_param2 = ops.parameter(PartialShape([2, 2]))
    model_add = ops.add(model_param1, model_param2)
    model_param3 = ops.parameter(PartialShape([2, 2]))
    model_mul = ops.matmul(model_add, model_param3, False, False)
    model_abs = ops.abs(model_mul)
    model_relu = ops.relu(model_abs)
    model_result = ops.result(model_relu) # noqa

    # Create a sample pattern an optional top node
    pattern_sig_opt = Optional(["opset13.Sigmoid"], AnyInput())
    pattern_mul = WrapType("opset13.MatMul", [pattern_sig_opt, AnyInput()])
    pattern_abs = WrapType("opset13.Abs", pattern_mul)
    pattern_relu = WrapType("opset13.Relu", pattern_abs)

    matcher = Matcher(pattern_relu, "FindPattern")

    # Should perfectly match even though there's no Sigmoid going into MatMul
    assert matcher.match(model_relu)


def test_pattern_optional_root():
    model_param1 = ops.parameter(PartialShape([2, 2]))
    model_param2 = ops.parameter(PartialShape([2, 2]))
    model_add = ops.add(model_param1, model_param2)
    model_param3 = ops.parameter(PartialShape([2, 2]))
    model_mul = ops.matmul(model_add, model_param3, False, False)
    model_abs = ops.abs(model_mul)
    model_relu = ops.relu(model_abs)
    model_result = ops.result(model_relu) # noqa

    # Create a sample pattern with an optional root node
    pattern_mul = WrapType("opset13.MatMul", [AnyInput(), AnyInput()])
    pattern_abs = WrapType("opset13.Abs", pattern_mul)
    pattern_relu = WrapType("opset13.Relu", pattern_abs)
    pattern_sig_opt = Optional(["opset13.Sigmoid"], pattern_relu)

    matcher = Matcher(pattern_sig_opt, "FindPattern")

    # Should perfectly match even though there's no Sigmoid as root
    assert matcher.match(model_relu)


def test_wrap_type_pattern_type():
    last_opset_number = 16
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


def test_any_input_symbol_predicate():
    def symbol_matching_test(shape: PartialShape, pattern: str):
        param = ops.parameter(shape)
        matcher = Matcher(AnyInput(shape_matches(pattern)), "Find" + pattern)
        assert matcher.match(param), f"Match failed for {shape} {pattern}"
        return matcher.get_symbols()

    symbols = symbol_matching_test(PartialShape([1, 3, 22, 22]), "[Batch,Channels,Spatial,Spatial]")
    assert symbols["Batch"] == 1, symbols
    assert symbols["Channels"] == 3, symbols
    assert symbols["Spatial"] == 22, symbols

    shape = PartialShape([-1, 2, 3, 4, -1, 6, 7])
    a_dim, b_dim = Dimension(), Dimension()
    a_dim.set_symbol(Symbol())
    b_dim.set_symbol(Symbol())
    shape[0] = a_dim
    shape[4] = b_dim
    symbols = symbol_matching_test(shape, "[Batches...,Dyn,Six,7]")
    assert symbols["Batches"] == [a_dim.get_symbol(), 2, 3, 4], symbols
    assert symbols["Dyn"] == b_dim.get_symbol(), symbols
    assert symbols["Six"] == 6, symbols


def test_optional_full_match():
    model_input = ops.parameter(PartialShape.dynamic())
    model_abs = ops.abs(model_input)
    model_relu = ops.relu(model_abs.output(0))

    pattern_abs = Optional(["opset13.Abs"])
    pattern_relu = ops.relu(pattern_abs.output(0))

    matcher = Matcher(pattern_relu, "FindRelu")
    assert matcher.match(model_relu)


def test_optional_one_node():
    model_input = ops.parameter(PartialShape.dynamic())
    model_relu = ops.relu(model_input)
    model_abs = ops.abs(model_input)

    assert Matcher(Optional(["opset13.Relu"]), "OneNodeTest").match(model_relu)
    assert not Matcher(Optional(["opset13.Abs"]), "OneNodeTest").match(model_relu)

    assert not Matcher(Optional(["opset13.Relu"]), "OneNodeTest").match(model_abs)

    assert Matcher(Optional(["opset13.Parameter"]), "OneNodeTest").match(ops.parameter(PartialShape.dynamic()))
    assert not Matcher(Optional(["opset13.Relu"]), "OneNodeTest").match(ops.parameter(PartialShape.dynamic()))


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


def test_optional_with_multi_input_node():
    model_input_0 = ops.parameter(PartialShape.dynamic())
    model_relu = ops.relu(model_input_0.output(0))
    model_input_1 = ops.parameter(PartialShape.dynamic())
    model_add = ops.add(model_relu, model_input_1)

    assert Matcher(Optional(["opset13.Add"], [model_relu, model_input_1]), "MultiInNode").match(model_add)
    assert Matcher(Optional(["opset13.Add"], [model_relu, model_input_1]), "MultiInNode").match(model_relu)
    assert not Matcher(Optional(["opset13.Add"], [model_relu, model_input_1]), "MultiInNode").match(model_input_1)
    assert not Matcher(Optional(["opset13.Add"], [model_relu, model_input_1]), "MultiInNode").match(model_input_0)

    assert not Matcher(Optional(["opset13.Add"], [model_relu, model_input_1], lambda x: False), "MultiInNodePredicate").match(model_add)
    assert Matcher(Optional(["opset13.Add"], [model_relu, model_input_1], lambda x: True), "MultiInNodePredicate").match(model_add)


def test_all_predicates():
    static_param = ops.parameter(PartialShape([1, 3, 22, 22]), np.float32)
    # np.compat.long =/= int:
    # https://numpy.org/devdocs/numpy_2_0_migration_guide.html#windows-default-integer
    dynamic_param = ops.parameter(PartialShape([-1, 6]), np.intp)
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
                                              get_element_type(np.intp)])), "Test").match(static_param)
    assert Matcher(WrapType("opset13.Parameter",  # noqa: ECE001
                            type_matches_any([get_element_type(np.float32),
                                              get_element_type(np.intp)])), "Test").match(dynamic_param)
