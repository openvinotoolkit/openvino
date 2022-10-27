# -*- coding: utf-8 -*-
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from functools import wraps
from typing import Callable, Any


def add_openvino_libs_to_path() -> None:
    """Adds OpenVINO libraries to the PATH environment variable on Windows."""
    if sys.platform == "win32":
        # Installer, yum, pip installs openvino dlls to the different directories
        # and those paths need to be visible to the openvino modules
        #
        # If you're using a custom installation of openvino,
        # add the location of openvino dlls to your system PATH.
        #
        # looking for the libs in the pip installation path by default.
        openvino_libs = [os.path.join(os.path.dirname(__file__), "libs")]
        # setupvars.bat script set all libs paths to OPENVINO_LIB_PATHS environment variable.
        openvino_libs_installer = os.getenv("OPENVINO_LIB_PATHS")
        if openvino_libs_installer:
            openvino_libs.extend(openvino_libs_installer.split(";"))
        for lib in openvino_libs:
            lib_path = os.path.join(os.path.dirname(__file__), lib)
            if os.path.isdir(lib_path):
                os.environ["PATH"] = os.path.abspath(lib_path) + ";" + os.environ["PATH"]
                # On Windows, with Python >= 3.8, DLLs are no longer imported from the PATH.
                if (3, 8) <= sys.version_info:
                    os.add_dll_directory(os.path.abspath(lib_path))


def deprecated(version: str = "", message: str = "") -> Callable[..., Any]:
    """Prints deprecation warning "{function_name} is deprecated and will be removed in version {version}. {message}" and runs the function.

    :param version: The version in which the code will be removed.
    :param message: A message explaining why the function is deprecated and/or what to use instead.
    """
    def decorator(wrapped: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(wrapped)
        def wrapper(*args: Any, **kwargs: Any) -> Callable[..., Any]:
            # it must be imported here; otherwise, there are errors with no loaded DLL for Windows
            from openvino._pyopenvino.util import deprecation_warning
            deprecation_warning(wrapped.__name__, version, message)
            return wrapped(*args, **kwargs)
        return wrapper
    return decorator
