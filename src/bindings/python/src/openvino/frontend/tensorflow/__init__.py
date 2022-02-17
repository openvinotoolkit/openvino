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
    from openvino.frontend.tensorflow.py_tensorflow_frontend import ConversionExtensionTensorflow as ConversionExtension
    from openvino.frontend.tensorflow.py_tensorflow_frontend import OpExtensionTensorflow as OpExtension
except ImportError as err:
    raise ImportError("OpenVINO Tensorflow frontend is not available, please make sure the frontend is built. "
                      "{}".format(err))
