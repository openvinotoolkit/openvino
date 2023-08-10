# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openvino module namespace, exposing factory functions for all ops and other classes."""
# noqa: F401

from openvino.utils import _add_openvino_libs_to_search_path

_add_openvino_libs_to_search_path()

from openvino._pyopenvino import get_version

__version__ = get_version()

# Openvino pybind bindings and python extended classes
from openvino._pyopenvino import Dimension
from openvino._pyopenvino import Input
from openvino._pyopenvino import Output
from openvino._pyopenvino import Node
from openvino._pyopenvino import Type
from openvino._pyopenvino import PartialShape
from openvino._pyopenvino import Shape
from openvino._pyopenvino import Strides
from openvino._pyopenvino import CoordinateDiff
from openvino._pyopenvino import DiscreteTypeInfo
from openvino._pyopenvino import AxisSet
from openvino._pyopenvino import AxisVector
from openvino._pyopenvino import Coordinate
from openvino._pyopenvino import Layout
from openvino._pyopenvino import ConstOutput
from openvino._pyopenvino import layout_helpers
from openvino._pyopenvino import OVAny
from openvino._pyopenvino import RTMap
from openvino.runtime.ie_api import Core
from openvino.runtime.ie_api import CompiledModel
from openvino.runtime.ie_api import InferRequest
from openvino.runtime.ie_api import Model
from openvino.runtime.ie_api import AsyncInferQueue
from openvino._pyopenvino import Version
from openvino._pyopenvino import Tensor
from openvino._pyopenvino import Extension
from openvino._pyopenvino import ProfilingInfo
from openvino._pyopenvino import get_batch
from openvino._pyopenvino import set_batch
from openvino._pyopenvino import serialize
from openvino._pyopenvino import save_model
from openvino._pyopenvino import shutdown

# Import opsets
from openvino.runtime import opset1
from openvino.runtime import opset2
from openvino.runtime import opset3
from openvino.runtime import opset4
from openvino.runtime import opset5
from openvino.runtime import opset6
from openvino.runtime import opset7
from openvino.runtime import opset8
from openvino.runtime import opset9
from openvino.runtime import opset10
from openvino.runtime import opset11
from openvino.runtime import opset12

# Import properties API
from openvino.runtime import properties

# Helper functions for openvino module
from openvino.runtime.ie_api import tensor_from_file
from openvino.runtime.ie_api import compile_model

# Extend Node class to support binary operators
Node.__add__ = opset12.add
Node.__sub__ = opset12.subtract
Node.__mul__ = opset12.multiply
Node.__div__ = opset12.divide
Node.__truediv__ = opset12.divide
Node.__radd__ = lambda left, right: opset12.add(right, left)
Node.__rsub__ = lambda left, right: opset12.subtract(right, left)
Node.__rmul__ = lambda left, right: opset12.multiply(right, left)
Node.__rdiv__ = lambda left, right: opset12.divide(right, left)
Node.__rtruediv__ = lambda left, right: opset12.divide(right, left)
Node.__eq__ = opset12.equal
Node.__ne__ = opset12.not_equal
Node.__lt__ = opset12.less
Node.__le__ = opset12.less_equal
Node.__gt__ = opset12.greater
Node.__ge__ = opset12.greater_equal
