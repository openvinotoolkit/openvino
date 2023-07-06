# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.ovc.convert import convert_model, InputCutInfo, LayoutMap

try:
    import openvino.runtime  # pylint: disable=no-name-in-module,import-error
    openvino.runtime.convert_model = convert_model  # pylint: disable=no-name-in-module,no-member
    openvino.runtime.InputCutInfo = InputCutInfo  # pylint: disable=no-name-in-module,no-member
    openvino.runtime.LayoutMap = LayoutMap  # pylint: disable=no-name-in-module,no-member
except:
    pass