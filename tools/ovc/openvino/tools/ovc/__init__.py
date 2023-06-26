# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.ovc.convert import convert_model, InputCutInfo, LayoutMap

try:
    import openvino.runtime
    openvino.runtime.convert_model = convert_model
    openvino.runtime.InputCutInfo = InputCutInfo
    openvino.runtime.LayoutMap = LayoutMap
except:
    pass