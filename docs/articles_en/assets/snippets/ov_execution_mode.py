# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#! [ov:execution_mode:part0]
import openvino as ov
import openvino.properties.hint as hints

core = ov.Core()
# in case of Accuracy
core.set_property(
    "CPU",
    {hints.execution_mode: hints.ExecutionMode.ACCURACY},
)
# in case of Performance
core.set_property(
    "CPU",
    {hints.execution_mode: hints.ExecutionMode.PERFORMANCE},
)
#! [ov:execution_mode:part0]
