# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openvino module namespace, exposing factory functions for all ops and other classes."""
# noqa: F401

from pkg_resources import get_distribution, DistributionNotFound

__path__ = __import__('pkgutil').extend_path(__path__, __name__)  # type: ignore  # mypy issue #1422

try:
    __version__ = get_distribution("openvino-core").version
except DistributionNotFound:
    __version__ = "0.0.0.dev0"

from openvino.ie_api import BlobWrapper
from openvino.ie_api import infer
from openvino.ie_api import async_infer
from openvino.ie_api import get_result
from openvino.ie_api import blob_from_file

from openvino.impl import Dimension
from openvino.impl import Function
from openvino.impl import Node
from openvino.impl import PartialShape
from openvino.impl import Layout

from openvino.pyopenvino import Core
from openvino.pyopenvino import IENetwork
from openvino.pyopenvino import ExecutableNetwork
from openvino.pyopenvino import Version
from openvino.pyopenvino import Parameter
from openvino.pyopenvino import InputInfoPtr
from openvino.pyopenvino import InputInfoCPtr
from openvino.pyopenvino import DataPtr
from openvino.pyopenvino import TensorDesc
from openvino.pyopenvino import get_version
from openvino.pyopenvino import StatusCode
from openvino.pyopenvino import InferQueue
from openvino.pyopenvino import InferRequest  # TODO: move to ie_api?
from openvino.pyopenvino import Blob
from openvino.pyopenvino import PreProcessInfo
from openvino.pyopenvino import MeanVariant
from openvino.pyopenvino import ResizeAlgorithm
from openvino.pyopenvino import ColorFormat
from openvino.pyopenvino import PreProcessChannel
from openvino.pyopenvino import Tensor

from openvino import opset1
from openvino import opset2
from openvino import opset3
from openvino import opset4
from openvino import opset5
from openvino import opset6
from openvino import opset7
from openvino import opset8

# Extend Node class to support binary operators
Node.__add__ = opset8.add
Node.__sub__ = opset8.subtract
Node.__mul__ = opset8.multiply
Node.__div__ = opset8.divide
Node.__truediv__ = opset8.divide
Node.__radd__ = lambda left, right: opset8.add(right, left)
Node.__rsub__ = lambda left, right: opset8.subtract(right, left)
Node.__rmul__ = lambda left, right: opset8.multiply(right, left)
Node.__rdiv__ = lambda left, right: opset8.divide(right, left)
Node.__rtruediv__ = lambda left, right: opset8.divide(right, left)
Node.__eq__ = opset8.equal
Node.__ne__ = opset8.not_equal
Node.__lt__ = opset8.less
Node.__le__ = opset8.less_equal
Node.__gt__ = opset8.greater
Node.__ge__ = opset8.greater_equal

# Patching for Blob class
# flake8: noqa: F811
# this class will be removed
Blob = BlobWrapper
# Patching ExecutableNetwork
ExecutableNetwork.infer = infer
# Patching InferRequest
InferRequest.infer = infer
InferRequest.async_infer = async_infer
InferRequest.get_result = get_result
# Patching InferQueue
InferQueue.async_infer = async_infer
