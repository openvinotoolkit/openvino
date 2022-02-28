# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Package: openvino
Low level wrappers for the FrontEnd C++ API.
"""

# flake8: noqa

from openvino.utils import add_openvino_libs_to_path

add_openvino_libs_to_path()

try:
    from openvino.frontend.onnx.py_onnx_frontend import ConversionExtensionONNX as ConversionExtension
    from openvino.frontend.onnx.py_onnx_frontend import OpExtensionONNX as OpExtension
except ImportError as err:
    raise ImportError("OpenVINO ONNX frontend is not available, please make sure the frontend is built. "
                      "{}".format(err))
