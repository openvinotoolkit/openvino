# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

# Required for Windows OS platforms
# Note: always top-level
try:
    from openvino.build_utils import _add_openvino_libs_to_search_path
    _add_openvino_libs_to_search_path()
except ImportError:
    pass

# #
# # OpenVINO API
# # This __init__.py forces checking of runtime modules to propagate errors.
# # It is not compared with init files from openvino-dev package.
# #

# Openvino pybind bindings 
from openvino._pyopenvino import AxisSet
from openvino._pyopenvino import AxisVector
from openvino._pyopenvino import ConstOutput
from openvino._pyopenvino import Coordinate
from openvino._pyopenvino import CoordinateDiff
from openvino._pyopenvino import DiscreteTypeInfo
from openvino._pyopenvino import Extension
from openvino._pyopenvino import ProfilingInfo
from openvino._pyopenvino import RTMap
from openvino._pyopenvino import Version
from openvino._pyopenvino import Symbol
from openvino._pyopenvino import Dimension
from openvino._pyopenvino import Input
from openvino._pyopenvino import Output
from openvino._pyopenvino import Node
from openvino._pyopenvino import Strides
from openvino._pyopenvino import PartialShape
from openvino._pyopenvino import Shape
from openvino._pyopenvino import Layout
from openvino._pyopenvino import Type
from openvino._pyopenvino import Tensor
from openvino._pyopenvino import OVAny
from openvino._pyopenvino import get_batch
from openvino._pyopenvino import set_batch
from openvino._pyopenvino import serialize
from openvino._pyopenvino import shutdown
from openvino._pyopenvino import save_model
from openvino._pyopenvino import layout_helpers
from openvino._pyopenvino import RemoteContext
from openvino._pyopenvino import RemoteTensor
from openvino._pyopenvino import Op

# Python extended classes and functions
from openvino.ie_api import Model
from openvino.ie_api import Core
from openvino.ie_api import CompiledModel
from openvino.ie_api import InferRequest
from openvino.ie_api import AsyncInferQueue
from openvino.ie_api import tensor_from_file
from openvino.ie_api import compile_model

# libva related:
from openvino._pyopenvino import VAContext
from openvino._pyopenvino import VASurfaceTensor

# Set version for openvino package
from openvino._pyopenvino import get_version
__version__ = get_version()

# Tools
try:
    # Model Conversion API - ovc should reside in the main namespace
    from openvino.tools.ovc import convert_model
except ImportError:
    pass
