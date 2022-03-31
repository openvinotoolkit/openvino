# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from openvino.runtime import PartialShape, opset8
from openvino.runtime.passes import Matcher, WrapType, Or, AnyInput
from openvino.runtime.passes import consumers_count, has_static_dim, has_static_dims, \
    has_static_shape, has_static_rank, type_matches, type_matches_any
from openvino.runtime.utils.types import get_element_type

from utils.utils import expect_exception


def test_wrap_type_pattern_type():
    for i in range(1, 9):
        WrapType("opset{}.Parameter".format(i))
        WrapType("opset{}::Parameter".format(i))

    # Negative check not to forget to update opset map in get_type function
    expect_exception(lambda: WrapType("opset9.Parameter"), "Unsupported opset type: opset9")

    # Generic negative test cases
    expect_exception(lambda: WrapType(""))
    expect_exception(lambda: WrapType("opset8"))
    expect_exception(lambda: WrapType("Parameter"))
    expect_exception(lambda: WrapType("opset.Parameter"))
    expect_exception(lambda: WrapType("opset8,Parameter"))
    expect_exception(lambda: WrapType("Parameter.opset8"))


def test_wrap_type_ctors():
    param = opset8.parameter(PartialShape([1, 3, 22, 22]))
    relu = opset8.relu(param.output(0))
    slope = opset8.parameter(PartialShape([]))
    prelu = opset8.prelu(param.output(0), slope.output(0))

    m = Matcher(WrapType(["opset8.Relu", "opset8.PRelu"]), "FindActivation")
    assert m.match(relu)
    assert m.match(prelu)

    m = Matcher(WrapType(["opset8.Relu", "opset8.PRelu"],
                WrapType("opset8.Parameter").output(0)), "FindActivation")
    assert m.match(relu)


def test_or():
    param = opset8.parameter(PartialShape([1, 3, 22, 22]))
    relu = opset8.relu(param.output(0))
    slope = opset8.parameter(PartialShape([]))
    prelu = opset8.prelu(param.output(0), slope.output(0))

    m = Matcher(Or([WrapType("opset8.Relu"),
                    WrapType("opset8.PRelu")]), "FindActivation")
    assert m.match(relu)
    assert m.match(prelu)


def test_any_input():
    param = opset8.parameter(PartialShape([1, 3, 22, 22]))
    relu = opset8.relu(param.output(0))
    slope = opset8.parameter(PartialShape([]))
    prelu = opset8.prelu(param.output(0), slope.output(0))

    m = Matcher(WrapType("opset8.PRelu", [AnyInput(), AnyInput()]), "FindActivation")
    assert not m.match(relu)
    assert m.match(prelu)


def test_any_input_predicate():
    param = opset8.parameter(PartialShape([1, 3, 22, 22]))
    slope = opset8.parameter(PartialShape([]))

    m = Matcher(AnyInput(lambda output: len(output.get_shape()) == 4), "FindActivation")
    assert m.match(param)
    assert not m.match(slope)


def test_all_predicates():
    static_param = opset8.parameter(PartialShape([1, 3, 22, 22]), np.float32)
    dynamic_param = opset8.parameter(PartialShape([-1, 6]), np.long)
    fully_dynamic_param = opset8.parameter(PartialShape.dynamic())

    assert Matcher(WrapType("opset8.Parameter", consumers_count(0)), "Test").match(static_param)
    assert not Matcher(WrapType("opset8.Parameter", consumers_count(1)), "Test").match(static_param)

    assert Matcher(WrapType("opset8.Parameter", has_static_dim(1)), "Test").match(static_param)
    assert not Matcher(WrapType("opset8.Parameter", has_static_dim(0)), "Test").match(dynamic_param)

    assert Matcher(WrapType("opset8.Parameter", has_static_dims([0, 3])), "Test").match(static_param)
    assert not Matcher(WrapType("opset8.Parameter", has_static_dims([0, 1])), "Test").match(dynamic_param)

    assert Matcher(WrapType("opset8.Parameter", has_static_shape()), "Test").match(static_param)
    assert not Matcher(WrapType("opset8.Parameter", has_static_shape()), "Test").match(dynamic_param)

    assert Matcher(WrapType("opset8.Parameter", has_static_rank()), "Test").match(dynamic_param)
    assert not Matcher(WrapType("opset8.Parameter", has_static_rank()), "Test").match(fully_dynamic_param)

    assert Matcher(WrapType("opset8.Parameter",
                            type_matches(get_element_type(np.float32))), "Test").match(static_param)
    assert not Matcher(WrapType("opset8.Parameter",
                                type_matches(get_element_type(np.float32))), "Test").match(dynamic_param)

    assert Matcher(WrapType("opset8.Parameter",
                            type_matches_any([get_element_type(np.float32),
                                              get_element_type(np.long)])), "Test").match(static_param)
    assert Matcher(WrapType("opset8.Parameter",
                            type_matches_any([get_element_type(np.float32),
                                              get_element_type(np.long)])), "Test").match(dynamic_param)
