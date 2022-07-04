# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openvino module namespace, exposing factory functions for all ops and other classes."""
# noqa: F401

from openvino.utils import add_openvino_libs_to_path
from pkg_resources import get_distribution, DistributionNotFound


try:
    __version__ = get_distribution("openvino-core").version
except DistributionNotFound:
    __version__ = "0.0.0.dev0"

add_openvino_libs_to_path()

# Openvino pybind bindings and python extended classes
from openvino.pyopenvino import Dimension
from openvino.pyopenvino import Model
from openvino.pyopenvino import Input
from openvino.pyopenvino import Output
from openvino.pyopenvino import Node
from openvino.pyopenvino import Type
from openvino.pyopenvino import PartialShape
from openvino.pyopenvino import Shape
from openvino.pyopenvino import Strides
from openvino.pyopenvino import CoordinateDiff
from openvino.pyopenvino import DiscreteTypeInfo
from openvino.pyopenvino import AxisSet
from openvino.pyopenvino import AxisVector
from openvino.pyopenvino import Coordinate
from openvino.pyopenvino import Layout
from openvino.pyopenvino import ConstOutput
from openvino.pyopenvino import layout_helpers
from openvino.pyopenvino import OVAny
from openvino.pyopenvino import RTMap
from openvino.runtime.ie_api import Core
from openvino.runtime.ie_api import CompiledModel
from openvino.runtime.ie_api import InferRequest
from openvino.runtime.ie_api import AsyncInferQueue
from openvino.pyopenvino import Version
from openvino.pyopenvino import Tensor
from openvino.pyopenvino import Extension
from openvino.pyopenvino import ProfilingInfo
from openvino.pyopenvino import get_version
from openvino.pyopenvino import get_batch
from openvino.pyopenvino import set_batch
from openvino.pyopenvino import serialize

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

# Import properties API
from openvino.pyopenvino import properties

# Helper functions for openvino module
from openvino.runtime.ie_api import tensor_from_file
from openvino.runtime.ie_api import compile_model


# Extend Node class to support binary operators
Node.__add__ = opset9.add
Node.__sub__ = opset9.subtract
Node.__mul__ = opset9.multiply
Node.__div__ = opset9.divide
Node.__truediv__ = opset9.divide
Node.__radd__ = lambda left, right: opset9.add(right, left)
Node.__rsub__ = lambda left, right: opset9.subtract(right, left)
Node.__rmul__ = lambda left, right: opset9.multiply(right, left)
Node.__rdiv__ = lambda left, right: opset9.divide(right, left)
Node.__rtruediv__ = lambda left, right: opset9.divide(right, left)
Node.__eq__ = opset9.equal
Node.__ne__ = opset9.not_equal
Node.__lt__ = opset9.less
Node.__le__ = opset9.less_equal
Node.__gt__ = opset9.greater
Node.__ge__ = opset9.greater_equal
