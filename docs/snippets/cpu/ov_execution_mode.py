# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino as ov

#! [ov:execution_mode:part0]
core = ov.Core()
# in case of Accuracy
core.set_property("CPU", {"EXECUTION_MODE_HINT": "ACCURACY"})
# in case of Performance
core.set_property("CPU", {"EXECUTION_MODE_HINT": "PERFORMANCE"})
#! [ov:execution_mode:part0]
