# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Enums
from openvino._pyopenvino.properties.intel_npu import CompilerType

# Properties
import openvino._pyopenvino.properties.intel_npu as __intel_npu
from openvino.properties._properties import __make_properties

__make_properties(__intel_npu, __name__)
