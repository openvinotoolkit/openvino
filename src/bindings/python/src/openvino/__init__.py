# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

__path__ = __import__("pkgutil").extend_path(__path__, __name__)  # type: ignore # mypy issue #1422

import warnings

# Required for Windows OS platforms
from .utils import _add_openvino_libs_to_search_path

_add_openvino_libs_to_search_path()

# Import openvino.runtime and most important classes
from . import runtime

from .runtime import get_version
__version__ = get_version()

from .runtime import Model
from .runtime import Core
from .runtime import AsyncInferQueue
from .runtime import Type
from .runtime import PartialShape
from .runtime import Shape
from .runtime import Strides
from .runtime import Layout
from .runtime import Tensor

from .runtime import get_batch
from .runtime import set_batch
from .runtime import serialize
from .runtime import shutdown
from .runtime import tensor_from_file
from .runtime import compile_model

# TODO should opsets go here?

# Import all additional modules
from . import frontend
from . import helpers
from . import preprocess

# Try to import openvino.tools
try:
    from . import tools
    from .tools.mo import convert_model  # TODO , InputCutInfo, LayoutMap
except ImportError:
    warnings.warn("openvino.tools module could not be found!", ImportWarning)
