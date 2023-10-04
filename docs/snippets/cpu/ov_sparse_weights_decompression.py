# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import openvino as ov
from snippets import get_model

model = get_model()

device_name = "CPU"
xml_path = "model.xml"
# ! [ov:intel_cpu:sparse_weights_decompression:part0]
core = ov.Core()
core.set_property("CPU", ov.properties.intel_cpu.sparse_weights_decompression_rate(0.8))
compiled_model = core.compile_model(model=model, device_name=device_name)
# ! [ov:intel_cpu:sparse_weights_decompression:part0]
assert compiled_model
