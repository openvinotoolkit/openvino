# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.runtime import Core

#! [ov:execution_mode:part0]
core = Core()
core.set_property("CPU", {"EXECUTION_MODE_HINT": "ACCURACY"})
#! [ov:execution_mode:part0]
