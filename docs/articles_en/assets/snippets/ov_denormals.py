# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import openvino as ov
from utils import get_model

device_name = 'CPU'
model = get_model()

# ! [ov:intel_cpu:denormals_optimization:part0]
import openvino.properties.intel_cpu as intel_cpu

core = ov.Core()
core.set_property("CPU", intel_cpu.denormals_optimization(True))
compiled_model = core.compile_model(model=model, device_name=device_name)
# ! [ov:intel_cpu:denormals_optimization:part0]
assert compiled_model
