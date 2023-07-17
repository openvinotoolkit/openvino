# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Required for Windows OS platforms
from openvino.utils import _add_openvino_libs_to_search_path

_add_openvino_libs_to_search_path()

# Import openvino.runtime and most important classes
from openvino import runtime

from openvino.runtime import get_version
__version__ = get_version()

from openvino.runtime import Model
from openvino.runtime import Core
from openvino.runtime import AsyncInferQueue
from openvino.runtime import Type
from openvino.runtime import PartialShape
from openvino.runtime import Shape
from openvino.runtime import Strides
from openvino.runtime import Layout
from openvino.runtime import Tensor

from openvino.runtime import get_batch
from openvino.runtime import set_batch
from openvino.runtime import serialize
from openvino.runtime import shutdown
from openvino.runtime import tensor_from_file
from openvino.runtime import compile_model

# TODO should opsets go here?

# Import all additional modules
from openvino import frontend as frontend
from openvino import helpers as helpers
from openvino import preprocess as preprocess

# Try to import openvino.tools
try:
    from openvino import tools
    from openvino.tools.mo import convert_model, InputCutInfo, LayoutMap
except ImportError:
    import warnings
    warnings.warn("openvino.tools module could not be found!", ImportWarning, stacklevel=2)
