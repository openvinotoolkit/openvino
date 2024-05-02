# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Package: openvino
Low level wrappers for the FrontEnd C++ API.
"""

# flake8: noqa

from openvino._pyopenvino import get_version

__version__ = get_version()

# exceptions
# extensions
from openvino._pyopenvino import (
    ConversionExtension,
    DecoderTransformationExtension,
    GeneralFailure,
    InitializationFailure,
    InputModel,
    NodeContext,
    NotImplementedFailure,
    OpConversionFailure,
    OpExtension,
    OpValidationFailure,
    Place,
    ProgressReporterExtension,
    TelemetryExtension,
)

# main classes
from openvino.frontend.frontend import FrontEnd, FrontEndManager
