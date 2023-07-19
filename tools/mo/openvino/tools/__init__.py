# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

try:
    import openvino.tools.mo
except ImportError:
    import warnings
    warnings.warn("openvino.tools.mo module could not be found!", ImportWarning, stacklevel=2)
