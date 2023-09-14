# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#! [ov:execution_mode:part0]
import openvino as ov

core = ov.Core()
# in case of Accuracy
core.set_property(
    "CPU",
    {ov.properties.hint.execution_mode(): ov.properties.hint.ExecutionMode.ACCURACY},
)
# in case of Performance
core.set_property(
    "CPU",
    {ov.properties.hint.execution_mode(): ov.properties.hint.ExecutionMode.PERFORMANCE},
)
#! [ov:execution_mode:part0]
