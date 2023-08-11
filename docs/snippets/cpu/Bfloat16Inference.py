# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import openvino as ov

#! [part0]
core = ov.Core()
cpu_optimization_capabilities = core.get_property("CPU", "OPTIMIZATION_CAPABILITIES")
#! [part0]

# TODO: enable part1 when property api will be supported in python
#! [part1]
core = ov.Core()
model = core.read_model("model.xml")
compiled_model = core.compile_model(model, "CPU")
inference_precision = core.get_property("CPU", "INFERENCE_PRECISION_HINT")
#! [part1]

#! [part2]
core = ov.Core()
core.set_property("CPU", {"INFERENCE_PRECISION_HINT": "f32"})
#! [part2]
