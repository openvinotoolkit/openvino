# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

try:
    import openvino.tools.mo
except ImportError:
    import warnings
    warnings.warn("openvino.tools.mo module could not be found!", ImportWarning, stacklevel=2)
