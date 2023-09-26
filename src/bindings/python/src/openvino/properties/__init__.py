# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Enums
from openvino._pyopenvino.properties import Affinity

# Properties
import openvino._pyopenvino.properties as __properties
from openvino.properties._properties import __make_properties
__make_properties(__properties, __name__)

# Submodules
from openvino.runtime.properties import hint
from openvino.runtime.properties import intel_cpu
from openvino.runtime.properties import intel_gpu
from openvino.runtime.properties import intel_auto
from openvino.runtime.properties import device
from openvino.runtime.properties import log
from openvino.runtime.properties import streams
