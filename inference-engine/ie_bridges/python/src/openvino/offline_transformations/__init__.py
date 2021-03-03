# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import sys

if sys.platform == "win32":
    # Installer, yum, pip installs openvino dlls to the different directory
    # and this path needs to be visible to the openvino modules
    #
    # If you're using a custom installation of openvino,
    # add the location of openvino dlls to your system PATH.
    openvino_libs = [
            '../../../../deployment_tools/inference_engine/bin/intel64/Release',
            '../../../../deployment_tools/inference_engine/bin/intel64/Debug',
            '../../../../deployment_tools/inference_engine/external/hddl/bin',
            '../../../../deployment_tools/inference_engine/external/gna/lib',
            '../../../../deployment_tools/inference_engine/external/tbb/bin',
            '../../../../deployment_tools/ngraph/lib',
            '../../openvino/libs',  # pip specific directory
        ]
    for lib in openvino_libs:
        lib_path = os.path.join(os.path.dirname(__file__), lib)
        if os.path.isdir(lib_path):
            # On Windows, with Python >= 3.8, DLLs are no longer imported from the PATH.
            if (3, 8) <= sys.version_info:
                os.add_dll_directory(os.path.abspath(lib_path))
            else:
                os.environ["PATH"] = os.path.abspath(lib_path) + ";" + os.environ["PATH"]

from .offline_transformations_api import *
__all__ = ['ApplyMOCTransformations']
