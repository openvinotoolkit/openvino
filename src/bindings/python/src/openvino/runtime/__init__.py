# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openvino module namespace, exposing factory functions for all ops and other classes."""
# noqa: F401

import warnings
warnings.filterwarnings("once", category=DeprecationWarning, module="openvino.runtime")
warnings.warn(
    "The `openvino.runtime` module is deprecated and will be removed in the 2026.0 release. "
    "Please replace `openvino.runtime` with `openvino`.",
    DeprecationWarning,
    stacklevel=1
)

from openvino import CompiledModel
