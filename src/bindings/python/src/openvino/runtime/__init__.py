# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openvino module namespace, exposing factory functions for all ops and other classes."""
# noqa: F401

import os
import sys

from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution("openvino-core").version
except DistributionNotFound:
    __version__ = "0.0.0.dev0"

if sys.platform == "win32":
    # Installer, yum, pip installs openvino dlls to the different directories
    # and those paths need to be visible to the openvino modules
    #
    # If you're using a custom installation of openvino,
    # add the location of openvino dlls to your system PATH.
    #
    # looking for the libs in the pip installation path by default.
    openvino_libs = [
        os.path.join(os.path.dirname(__file__), "..", "..", ".."),
        os.path.join(os.path.dirname(__file__), "..", "..", "openvino", "libs"),
    ]
    # setupvars.bat script set all libs paths to OPENVINO_LIB_PATHS environment variable.
    openvino_libs_installer = os.getenv("OPENVINO_LIB_PATHS")
    if openvino_libs_installer:
        openvino_libs.extend(openvino_libs_installer.split(";"))
    for lib in openvino_libs:
        lib_path = os.path.join(os.path.dirname(__file__), lib)
        if os.path.isdir(lib_path):
            # On Windows, with Python >= 3.8, DLLs are no longer imported from the PATH.
            if (3, 8) <= sys.version_info:
                os.add_dll_directory(os.path.abspath(lib_path))
            else:
                os.environ["PATH"] = (
                    os.path.abspath(lib_path) + ";" + os.environ["PATH"]
                )


# Openvino pybind bindings and python extended classes
from openvino.runtime.impl import Dimension
from openvino.runtime.impl import Function
from openvino.runtime.impl import Node
from openvino.runtime.impl import PartialShape
from openvino.runtime.impl import Layout

from openvino.runtime.ie_api import Core
from openvino.runtime.ie_api import ExecutableNetwork
from openvino.runtime.ie_api import InferRequest
from openvino.runtime.ie_api import AsyncInferQueue
from openvino.runtime.ie_api import Variant
from openvino.pyopenvino import Version
from openvino.pyopenvino import Parameter
from openvino.pyopenvino import Tensor
from openvino.pyopenvino import ProfilingInfo
from openvino.pyopenvino import get_version

# Import opsets
from openvino.runtime import opset1
from openvino.runtime import opset2
from openvino.runtime import opset3
from openvino.runtime import opset4
from openvino.runtime import opset5
from openvino.runtime import opset6
from openvino.runtime import opset7
from openvino.runtime import opset8

# Helper functions for openvino module
from openvino.runtime.ie_api import tensor_from_file
from openvino.runtime.ie_api import compile_model

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
