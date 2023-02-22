# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import openvino.runtime as ov

device_name = 'CPU'
xml_path = 'modelWithDenormals.xml'
# ! [ov:intel_cpu:denormals_optimization:part0]
core = ov.Core()
core.set_property("CPU", ov.properties.intel_cpu.denormals_optimization(True))
model = core.read_model(model=xml_path)
compiled_model = core.compile_model(model=model, device_name=device_name)
# ! [ov:intel_cpu:denormals_optimization:part0]
assert compiled_model
