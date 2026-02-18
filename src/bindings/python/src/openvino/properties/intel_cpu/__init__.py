# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Enums
from openvino._pyopenvino.properties.intel_cpu import TbbPartitioner

# Properties
import openvino._pyopenvino.properties.intel_cpu as __intel_cpu
from openvino.properties._properties import __make_properties
__make_properties(__intel_cpu, __name__)
