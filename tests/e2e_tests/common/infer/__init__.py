# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .dummy_infer_class import use_dummy
from .provider import StepProvider

try:
    from .common_inference import Infer
except ImportError as e:
    Infer = use_dummy('ie_sync', str(e))
