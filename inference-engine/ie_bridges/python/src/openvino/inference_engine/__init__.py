# -*- coding: utf-8 -*-
# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

if sys.platform == 'win32':
    # Installer, yum, pip installs openvino dlls to the different directories
    # and those paths need to be visible to the openvino modules
    #
    # If you're using a custom installation of openvino,
    # add the location of openvino dlls to your system PATH.
    #
    # looking for the libs in the pip installation path by default.
    openvino_libs = [os.path.join(os.path.dirname(__file__), '..', '..', 'openvino', 'libs')]
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
                os.environ['PATH'] = os.path.abspath(lib_path) + ';' + os.environ['PATH']

from .ie_api import *

__all__ = ['IENetwork', 'TensorDesc', 'IECore', 'Blob', 'PreProcessInfo', 'get_version']
__version__ = get_version()  # type: ignore
