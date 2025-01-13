# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import openvino as ov
import openvino.properties.device as device
import openvino.properties.hint as hints

from snippets import get_model

model = get_model()

#! [part0]
core = ov.Core()
cpu_optimization_capabilities = core.get_property("CPU", device.capabilities)
#! [part0]

#! [part1]
core = ov.Core()
compiled_model = core.compile_model(model, "CPU")
inference_precision = core.get_property("CPU", hints.inference_precision)
#! [part1]

#! [part2]
core = ov.Core()
core.set_property("CPU", {hints.inference_precision: ov.Type.f32})
#! [part2]
