# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import sys

if sys.platform == "win32":
    # PIP installs openvino dlls 3 directories above in openvino.libs by default
    # and this path needs to be visible to the openvino modules
    #
    # If you're using a custom installation of openvino,
    # add the location of openvino dlls to your system PATH.
    openvino_dlls = os.path.join(os.path.dirname(__file__), "..", "..", "openvino", "libs")
    if (3, 8) <= sys.version_info:
        # On Windows, with Python >= 3.8, DLLs are no longer imported from the PATH.
        os.add_dll_directory(os.path.abspath(openvino_dlls))
    else:
        os.environ["PATH"] = os.path.abspath(openvino_dlls) + ";" + os.environ["PATH"]

from .offline_transformations_api import *
__all__ = ['ApplyMOCTransformations']
