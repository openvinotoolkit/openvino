# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openvino module namespace, exposing factory functions for all ops and other classes."""
# noqa: F401

from openvino._pyopenvino import get_version

__version__ = get_version()

# Openvino pybind bindings and python extended classes
from openvino._pyopenvino import (
    AxisSet,
    AxisVector,
    ConstOutput,
    Coordinate,
    CoordinateDiff,
    Dimension,
    DiscreteTypeInfo,
    Extension,
    Input,
    Layout,
    Node,
    Output,
    OVAny,
    PartialShape,
    ProfilingInfo,
    RTMap,
    Shape,
    Strides,
    Symbol,
    Tensor,
    Type,
    Version,
    get_batch,
    layout_helpers,
    save_model,
    serialize,
    set_batch,
    shutdown,
)

# Import properties API
# Import opsets
from openvino.runtime import (
    opset1,
    opset2,
    opset3,
    opset4,
    opset5,
    opset6,
    opset7,
    opset8,
    opset9,
    opset10,
    opset11,
    opset12,
    opset13,
    properties,
)

# Helper functions for openvino module
from openvino.runtime.ie_api import (
    AsyncInferQueue,
    CompiledModel,
    Core,
    InferRequest,
    Model,
    compile_model,
    tensor_from_file,
)

# Extend Node class to support binary operators
Node.__add__ = opset13.add
Node.__sub__ = opset13.subtract
Node.__mul__ = opset13.multiply
Node.__div__ = opset13.divide
Node.__truediv__ = opset13.divide
Node.__radd__ = lambda left, right: opset13.add(right, left)
Node.__rsub__ = lambda left, right: opset13.subtract(right, left)
Node.__rmul__ = lambda left, right: opset13.multiply(right, left)
Node.__rdiv__ = lambda left, right: opset13.divide(right, left)
Node.__rtruediv__ = lambda left, right: opset13.divide(right, left)
Node.__eq__ = opset13.equal
Node.__ne__ = opset13.not_equal
Node.__lt__ = opset13.less
Node.__le__ = opset13.less_equal
Node.__gt__ = opset13.greater
Node.__ge__ = opset13.greater_equal
