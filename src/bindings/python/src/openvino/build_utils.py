# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path


def _add_openvino_libs_to_search_path() -> None:
    """Add OpenVINO libraries to the DLL search path on Windows."""
    if sys.platform == "win32":
        # Installer, yum, pip installs openvino dlls to the different directories
        # and those paths need to be visible to the openvino modules
        #
        # If you're using a custom installation of openvino,
        # add the location of openvino dlls to your system PATH.
        openvino_libs = []
        if os.path.isdir(os.path.join(os.path.dirname(__file__), "libs")):
            # looking for the libs in the pip installation path.
            openvino_libs.append(os.path.join(os.path.dirname(__file__), "libs"))
        elif os.path.isdir(os.path.join(os.path.dirname(__file__), "..", "..", "..", "Library", "bin")):
            # looking for the libs in the conda installation path
            openvino_libs.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "Library", "bin"))
        else:
            # setupvars.bat script set all libs paths to OPENVINO_LIB_PATHS environment variable.
            openvino_libs_installer = os.getenv("OPENVINO_LIB_PATHS")
            if openvino_libs_installer:
                openvino_libs.extend(openvino_libs_installer.split(";"))
            else:
                sys.exit("Error: Please set the OPENVINO_LIB_PATHS environment variable. "
                         "If you use an install package, please, run setupvars.bat")
        for lib in openvino_libs:
            lib_path = os.path.join(os.path.dirname(__file__), lib)
            if os.path.isdir(lib_path):
                # On Windows, with Python >= 3.8, DLLs are no longer imported from the PATH.
                os.add_dll_directory(os.path.abspath(lib_path))


def get_cmake_path() -> str:
    """Searches for the directory containing CMake files within the package install directory.

    :return: The path to the directory containing CMake files, if found. Otherwise, returns empty string.
    :rtype: str
    """
    package_path = Path(__file__).parent
    cmake_file = "OpenVINOConfig.cmake"

    for dirpath, _, filenames in os.walk(package_path):
        if cmake_file in filenames:
            return dirpath

    return ""
