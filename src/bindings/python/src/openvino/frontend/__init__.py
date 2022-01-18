# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Package: openvino
Low level wrappers for the FrontEnd c++ api.
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
from openvino.pyopenvino import FrontEndManager
from openvino.pyopenvino import FrontEnd
from openvino.pyopenvino import InputModel
from openvino.pyopenvino import Place
from openvino.pyopenvino import TelemetryExtension
from openvino.pyopenvino import DecoderTransformationExtension
from openvino.pyopenvino import JsonConfigExtension
from openvino.pyopenvino import ProgressReporterExtension

# exceptions
from openvino.pyopenvino import NotImplementedFailure
from openvino.pyopenvino import InitializationFailure
from openvino.pyopenvino import OpConversionFailure
from openvino.pyopenvino import OpValidationFailure
from openvino.pyopenvino import GeneralFailure
