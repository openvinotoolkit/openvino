# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Package: openvino.impl
Low level wrappers for the c++ api.
"""

# flake8: noqa

import os
import sys

if sys.platform == "win32":
    # Installer, yum, pip installs openvino dlls to the different directories
    # and those paths need to be visible to the openvino modules
    #
    # If you're using a custom installation of openvino,
    # add the location of openvino dlls to your system PATH.
    #
    # looking for the libs in the pip installation path by default.
    openvino_libs = [os.path.join(os.path.dirname(__file__), '..', '..', '..'),
                     os.path.join(os.path.dirname(__file__), '..', '..', 'openvino', 'libs')]
    # setupvars.bat script set all libs paths to OPENVINO_LIB_PATHS environment variable.
    openvino_libs_installer = os.getenv('OPENVINO_LIB_PATHS')
    if openvino_libs_installer:
        openvino_libs.extend(openvino_libs_installer.split(';'))
    for lib in openvino_libs:
        lib_path = os.path.join(os.path.dirname(__file__), lib)
        if os.path.isdir(lib_path):
            # On Windows, with Python >= 3.8, DLLs are no longer imported from the PATH.
            if (3, 8) <= sys.version_info:
                os.add_dll_directory(os.path.abspath(lib_path))
            else:
                os.environ["PATH"] = os.path.abspath(lib_path) + ";" + os.environ["PATH"]

from openvino.pyopenvino import Dimension
from openvino.pyopenvino import Function
from openvino.pyopenvino import Input
from openvino.pyopenvino import Output
from openvino.pyopenvino import Node
from openvino.pyopenvino import Type
from openvino.pyopenvino import PartialShape
from openvino.pyopenvino import Shape
from openvino.pyopenvino import Strides
from openvino.pyopenvino import CoordinateDiff
from openvino.pyopenvino import AxisSet
from openvino.pyopenvino import AxisVector
from openvino.pyopenvino import Coordinate
from openvino.pyopenvino import Output

from openvino.pyopenvino import util
