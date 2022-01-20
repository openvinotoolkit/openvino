# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Package: ngraph
Low level wrappers for the PrePostProcessing c++ api.
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
    openvino_libs = [os.path.join(os.path.dirname(__file__), "..", "..", ".."),
                     os.path.join(os.path.dirname(__file__), "..", "..", "openvino", "libs")]
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
                os.environ["PATH"] = os.path.abspath(lib_path) + ";" + os.environ["PATH"]


# main classes
from openvino.pyopenvino.preprocess import InputInfo
from openvino.pyopenvino.preprocess import OutputInfo
from openvino.pyopenvino.preprocess import InputTensorInfo
from openvino.pyopenvino.preprocess import OutputTensorInfo
from openvino.pyopenvino.preprocess import InputModelInfo
from openvino.pyopenvino.preprocess import OutputModelInfo
from openvino.pyopenvino.preprocess import PrePostProcessor
from openvino.pyopenvino.preprocess import PreProcessSteps
from openvino.pyopenvino.preprocess import PostProcessSteps
from openvino.pyopenvino.preprocess import ColorFormat
from openvino.pyopenvino.preprocess import ResizeAlgorithm
