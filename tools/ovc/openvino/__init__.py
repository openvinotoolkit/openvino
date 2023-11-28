# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
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
# # API 2.0
# # This __init__.py forces checking of runtime modules to propagate errors.
# # It is not compared with init files from openvino-dev package.
# #
# Import all public modules

# Import most important classes and functions from openvino.runtime

# Set version for openvino package
from openvino.runtime import get_version

__version__ = get_version()

# Tools
try:
    # Model Conversion API - ovc should reside in the main namespace
    from openvino.tools.ovc import convert_model
except ImportError:
    pass
