# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

# Required for Windows OS platforms
# Note: always top-level
try:
    from openvino.utils import _add_openvino_libs_to_search_path
    _add_openvino_libs_to_search_path()
except ImportError:
    pass

# #
# # OpenVINO API
# # This __init__.py forces checking of runtime modules to propagate errors.
# # It is not compared with init files from openvino-dev package.
# #
# Import all public modules
from openvino import runtime as runtime
from openvino import frontend as frontend
from openvino import helpers as helpers
from openvino import preprocess as preprocess
from openvino import utils as utils
from openvino import properties as properties

# Import most important classes and functions from openvino.runtime
from openvino.runtime import Model
from openvino.runtime import Core
from openvino.runtime import CompiledModel
from openvino.runtime import InferRequest
from openvino.runtime import AsyncInferQueue

from openvino.runtime import Dimension
from openvino.runtime import Strides
from openvino.runtime import PartialShape
from openvino.runtime import Shape
from openvino.runtime import Layout
from openvino.runtime import Type
from openvino.runtime import Tensor
from openvino.runtime import OVAny

from openvino.runtime import compile_model
from openvino.runtime import get_batch
from openvino.runtime import set_batch
from openvino.runtime import serialize
from openvino.runtime import shutdown
from openvino.runtime import tensor_from_file
from openvino.runtime import save_model
from openvino.runtime import layout_helpers

from openvino._pyopenvino import RemoteContext
from openvino._pyopenvino import RemoteTensor

# libva related:
from openvino._pyopenvino import VAContext
from openvino._pyopenvino import VASurfaceTensor

# Set version for openvino package
from openvino.runtime import get_version
__version__ = get_version()

# Tools
try:
    # Model Conversion API - ovc should reside in the main namespace
    from openvino.tools.ovc import convert_model
except ImportError:
    pass
