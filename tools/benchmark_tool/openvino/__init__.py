# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

try:
    # Internal modules
    from openvino import test_utils as test_utils
except ImportError:
    pass

try:
    # Import all public modules
    from openvino import runtime as runtime
    from openvino import frontend as frontend
    from openvino import helpers as helpers
    from openvino import preprocess as preprocess
    from openvino import utils as utils
    from openvino.runtime import properties as properties
    # Import old API
    # TODO: remove in 2024.0
    from openvino import inference_engine as inference_engine

    # Required for Windows OS platforms
    from openvino.utils import _add_openvino_libs_to_search_path

    _add_openvino_libs_to_search_path()

    # Import most important classes and functions from openvino.runtime
    from openvino.runtime import Model
    from openvino.runtime import Core
    from openvino.runtime import CompiledModel
    from openvino.runtime import InferRequest
    from openvino.runtime import AsyncInferQueue

    from openvino.runtime.op import Constant
    from openvino.runtime.op import Parameter

    from openvino.runtime import Extension
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

    # Set version for openvino package
    from openvino.runtime import get_version
    __version__ = get_version()
except ImportError:
    import warnings
    warnings.warn("openvino package has problems with imports!", ImportWarning, stacklevel=2)

# Import openvino.tools
# Capture it in try-except with pass so circular imports are allowed.
# TODO: restructure packages to remove WA
try:
    from openvino import tools as tools
    # Model Conversion API - ovc should reside in the main namespace
    from openvino.tools.ovc import convert_model, InputCutInfo
except ImportError:
    pass
