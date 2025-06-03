# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# ! [ov:imports]
import pytest
from openvino import PartialShape
from openvino.runtime import opset13 as ops
from openvino.runtime.passes import Matcher, WrapType, Or, AnyInput, Optional
# ! [ov:imports]
from openvino.runtime.passes import (
    consumers_count,
    has_static_dim,
    has_static_dims,
    has_static_shape,
    has_static_rank,
    type_matches,
    type_matches_any,
)

# ! [ov:create_simple_model_and_pattern]
def simple_model_and_pattern():
    # Create a sample model
    model_param1 = ops.parameter(PartialShape([2, 2]))
    model_param2 = ops.parameter(PartialShape([2, 2]))
    model_add = ops.add(model_param1, model_param2)
    model_param3 = ops.parameter(PartialShape([2, 2]))
    model_mul = ops.matmul(model_add, model_param3, False, False)
    model_abs = ops.abs(model_mul)
    model_relu = ops.relu(model_abs)
    model_result = ops.result(model_relu)

    # Create a sample pattern
    pattern_mul = ops.matmul(AnyInput(), AnyInput(), False, False)
    pattern_abs = ops.abs(pattern_mul)
    pattern_relu = ops.relu(pattern_abs)

    # Create a matcher and try to match the nodes
    matcher = Matcher(pattern_relu, "FindPattern")

    # Should perfectly match
    assert matcher.match(model_relu)
# ! [ov:create_simple_model_and_pattern]

# ! [ov:create_simple_model_and_pattern_wrap_type]
def simple_model_and_pattern_wrap_type():
    model_param1 = ops.parameter(PartialShape([2, 2]))
    model_param2 = ops.parameter(PartialShape([2, 2]))
    model_add = ops.add(model_param1, model_param2)
    model_param3 = ops.parameter(PartialShape([2, 2]))
    model_mul = ops.matmul(model_add, model_param3, False, False)
    model_abs = ops.abs(model_mul)
    model_relu = ops.relu(model_abs)
    model_result = ops.result(model_relu)

    # Create a sample pattern
    pattern_mul = WrapType("opset13.MatMul", [AnyInput(), AnyInput()])
    pattern_abs = WrapType("opset13.Abs", pattern_mul)
    pattern_relu = WrapType("opset13.Relu", pattern_abs)

    # Create a matcher and try to match the nodes
    matcher = Matcher(pattern_relu, "FindPattern")

    # Should perfectly match
    assert matcher.match(model_relu)
# ! [ov:create_simple_model_and_pattern_wrap_type]

# ! [ov:wrap_type_list]
def wrap_type_list():
    model_param1 = ops.parameter(PartialShape([2, 2]))
    model_param2 = ops.parameter(PartialShape([2, 2]))
    model_add = ops.add(model_param1, model_param2)
    model_param3 = ops.parameter(PartialShape([2, 2]))
    model_mul = ops.matmul(model_add, model_param3, False, False)
    model_abs = ops.abs(model_mul)
    model_relu = ops.relu(model_abs)
    model_result = ops.result(model_relu)
    model_sig = ops.sigmoid(model_abs) # Note that we've added a Sigmoid node after Abs
    model_result1 = ops.result(model_sig)

    # Create a sample pattern
    pattern_mul = WrapType("opset13.MatMul", [AnyInput(), AnyInput()])
    pattern_abs = WrapType("opset13.Abs", pattern_mul)
    pattern_relu = WrapType(["opset13.Relu", "opset13.Sigmoid"], pattern_abs)

    # Create a matcher and try to match the nodes
    matcher = Matcher(pattern_relu, "FindPattern")

    # The same pattern perfectly matches 2 different nodes
    assert matcher.match(model_relu)
    assert matcher.match(model_sig)
# ! [ov:wrap_type_list]

def any_input():
# ! [ov:any_input]
    # Create a pattern with a MatMul node taking any inputs.
    pattern_mul = WrapType("opset13.MatMul", [AnyInput(), AnyInput()])
    pattern_abs = WrapType("opset13.Abs", pattern_mul)
    pattern_relu = WrapType("opset13.Relu", pattern_abs)
# ! [ov:any_input]

def wrap_type_predicate():
# ! [ov:wrap_type_predicate]
    WrapType("opset13.Relu", AnyInput(), consumers_count(2))
# ! [ov:wrap_type_predicate]

# ! [ov:any_input_predicate]
    # Create a pattern with an MatMul node taking any input that has a rank 4.
    pattern_mul = WrapType("opset13.MatMul", [AnyInput(lambda output: len(output.get_shape()) == 4), AnyInput(lambda output: len(output.get_shape()) == 4)])
    pattern_abs = WrapType("opset13.Abs", pattern_mul)
    pattern_relu = WrapType("opset13.Relu", pattern_abs)
# ! [ov:any_input_predicate]

# ! [ov:pattern_or]
def pattern_or():
    model_param1 = ops.parameter(PartialShape([2, 2]))
    model_param2 = ops.parameter(PartialShape([2, 2]))
    model_add = ops.add(model_param1, model_param2)
    model_param3 = ops.parameter(PartialShape([2, 2]))
    model_mul = ops.matmul(model_add, model_param3, False, False)
    model_abs = ops.abs(model_mul)
    model_relu = ops.relu(model_abs)
    model_result = ops.result(model_relu)

    # Create a red branch
    red_pattern_add = WrapType("opset13.Add", [AnyInput(), AnyInput()])
    red_pattern_relu = WrapType("opset13.Relu", red_pattern_add)
    red_pattern_sigmoid = WrapType(["opset13.Sigmoid"], red_pattern_relu)

    # Create a blue branch
    blue_pattern_mul = WrapType("opset13.MatMul", [AnyInput(), AnyInput()])
    blue_pattern_abs = WrapType("opset13.Abs", blue_pattern_mul)
    blue_pattern_relu = WrapType(["opset13.Relu"], blue_pattern_abs)

    #Create Or node
    pattern_or = Or([red_pattern_sigmoid, blue_pattern_relu])

    # Create a matcher and try to match the nodes
    matcher = Matcher(pattern_or, "FindPattern")

    # The same pattern perfectly matches 2 different nodes
    assert matcher.match(model_relu)
# ! [ov:pattern_or]

# ! [ov:pattern_optional_middle]
def pattern_optional_middle():
    model_param1 = ops.parameter(PartialShape([2, 2]))
    model_param2 = ops.parameter(PartialShape([2, 2]))
    model_add = ops.add(model_param1, model_param2)
    model_param3 = ops.parameter(PartialShape([2, 2]))
    model_mul = ops.matmul(model_add, model_param3, False, False)
    model_abs = ops.abs(model_mul)
    model_relu = ops.relu(model_abs)
    model_result = ops.result(model_relu)

    # Create a sample pattern with an Optional node in the middle
    pattern_mul = WrapType("opset13.MatMul", [AnyInput(), AnyInput()])
    pattern_abs = WrapType("opset13.Abs", pattern_mul)
    pattern_sig_opt = Optional(["opset13.Sigmoid"], pattern_abs)
    pattern_relu = WrapType("opset13.Relu", pattern_sig_opt)

    # Create a matcher and try to match the nodes
    matcher = Matcher(pattern_relu, "FindPattern")

    # Should perfectly match
    assert matcher.match(model_relu)
# ! [ov:pattern_optional_middle]

# ! [ov:pattern_optional_top]
def pattern_optional_top():
    model_param1 = ops.parameter(PartialShape([2, 2]))
    model_param2 = ops.parameter(PartialShape([2, 2]))
    model_add = ops.add(model_param1, model_param2)
    model_param3 = ops.parameter(PartialShape([2, 2]))
    model_mul = ops.matmul(model_add, model_param3, False, False)
    model_abs = ops.abs(model_mul)
    model_relu = ops.relu(model_abs)
    model_result = ops.result(model_relu)

    # Create a sample pattern an optional top node
    pattern_sig_opt = Optional(["opset13.Sigmoid"], AnyInput())
    pattern_mul = WrapType("opset13.MatMul", [pattern_sig_opt, AnyInput()])
    pattern_abs = WrapType("opset13.Abs", pattern_mul)
    pattern_relu = WrapType("opset13.Relu", pattern_abs)

    matcher = Matcher(pattern_relu, "FindPattern")

    # Should perfectly match even though there's no Sigmoid going into MatMul
    assert matcher.match(model_relu)
# ! [ov:pattern_optional_top]

# ! [ov:pattern_optional_root]
def pattern_optional_root():
    model_param1 = ops.parameter(PartialShape([2, 2]))
    model_param2 = ops.parameter(PartialShape([2, 2]))
    model_add = ops.add(model_param1, model_param2)
    model_param3 = ops.parameter(PartialShape([2, 2]))
    model_mul = ops.matmul(model_add, model_param3, False, False)
    model_abs = ops.abs(model_mul)
    model_relu = ops.relu(model_abs)
    model_result = ops.result(model_relu)

    # Create a sample pattern
    pattern_mul = WrapType("opset13.MatMul", [AnyInput(), AnyInput()])
    pattern_abs = WrapType("opset13.Abs", pattern_mul)
    pattern_relu = WrapType("opset13.Relu", pattern_abs)
    pattern_sig_opt = Optional(["opset13.Sigmoid"], pattern_relu)

    matcher = Matcher(pattern_sig_opt, "FindPattern")

    # Should perfectly match even though there's no Sigmoid as root
    assert matcher.match(model_relu)
# ! [ov:pattern_optional_root]

# ! [ov:optional_predicate]
    pattern_sig_opt = Optional(["opset13.Sigmoid"], pattern_relu, consumers_count(1))
# ! [ov:optional_predicate]
