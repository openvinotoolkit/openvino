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

# OpenVINO API
try:
    # Import all public modules
    # libva related:
    from openvino._pyopenvino import (
        RemoteContext,
        RemoteTensor,
        VAContext,
        VASurfaceTensor,
    )

    # Set version for openvino package
    # Import most important classes and functions from openvino.runtime
    from openvino.runtime import (
        AsyncInferQueue,
        CompiledModel,
        Core,
        Dimension,
        InferRequest,
        Layout,
        Model,
        OVAny,
        PartialShape,
        Shape,
        Strides,
        Symbol,
        Tensor,
        Type,
        compile_model,
        get_batch,
        get_version,
        layout_helpers,
        save_model,
        serialize,
        set_batch,
        shutdown,
        tensor_from_file,
    )

    from openvino import frontend as frontend
    from openvino import helpers as helpers
    from openvino import preprocess as preprocess
    from openvino import properties as properties
    from openvino import runtime as runtime
    from openvino import utils as utils

    __version__ = get_version()
except ImportError:
    import warnings

    warnings.warn(
        "openvino package has problems with imports!", ImportWarning, stacklevel=2
    )

# Tools
try:
    # Model Conversion API - ovc should reside in the main namespace
    from openvino.tools.ovc import convert_model
except ImportError:
    pass
