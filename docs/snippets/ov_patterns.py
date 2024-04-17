# Copyright (C) 2018-2024 Intel Corporation
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
    model_abs = ops.abs(model_add)
    model_relu = ops.relu(model_abs)

    # Create a sample pattern
    pattern_abs = ops.abs(AnyInput())
    pattern_relu = ops.relu(pattern_abs)

    # Create a matcher and try to match the nodes
    matcher = Matcher(pattern_relu, "FindPattern")

    # Should perfectly match
    assert matcher.match(model_relu)

# ! [ov:create_simple_model_and_pattern]

# ! [ov:create_simple_model_and_pattern_wrap_type]
def simple_model_and_pattern_wrap_type():
    # Create a sample model
    model_param1 = ops.parameter(PartialShape([2, 2]))
    model_param2 = ops.parameter(PartialShape([2, 2]))
    model_add = ops.add(model_param1, model_param2)
    model_abs = ops.abs(model_add)
    model_relu = ops.relu(model_abs)

    # Create a sample pattern
    pattern_abs = WrapType("opset13.Abs", AnyInput())
    pattern_relu = WrapType("opset13.Relu", pattern_abs)

    # Create a matcher and try to match the nodes
    matcher = Matcher(pattern_relu, "FindPattern")

    # Should perfectly match
    assert matcher.match(model_relu)

# ! [ov:create_simple_model_and_pattern_wrap_type]