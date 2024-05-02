# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Properties
import openvino._pyopenvino.properties as __properties

# Enums
from openvino._pyopenvino.properties import Affinity, CacheMode
from openvino.properties._properties import __make_properties

__make_properties(__properties, __name__)

# Submodules
from openvino.properties import (
    device,
    hint,
    intel_auto,
    intel_cpu,
    intel_gpu,
    log,
    streams,
)
