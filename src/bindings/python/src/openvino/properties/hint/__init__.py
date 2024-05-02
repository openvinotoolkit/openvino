# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Properties
import openvino._pyopenvino.properties.hint as __hint

# Enums
from openvino._pyopenvino.properties.hint import (
    ExecutionMode,
    ModelDistributionPolicy,
    PerformanceMode,
    Priority,
    SchedulingCoreType,
)
from openvino.properties._properties import __make_properties

__make_properties(__hint, __name__)
