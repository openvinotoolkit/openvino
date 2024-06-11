# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from openvino import Core, properties
from snippets import get_model

model = get_model()

device_name = "CPU"
core = Core()
core.set_property("CPU", properties.intel_cpu.sparse_weights_decompression_rate(0.8))

# ! [ov:intel_cpu:multi_threading:part0]
# Use one logical processor for inference
compiled_model_1 = core.compile_model(
    model=model,
    device_name=device_name,
    config={properties.inference_num_threads(): 1},
)

# Use logical processors of Efficient-cores for inference on hybrid platform
compiled_model_2 = core.compile_model(
    model=model,
    device_name=device_name,
    config={
        properties.hint.scheduling_core_type(): properties.hint.SchedulingCoreType.ECORE_ONLY,
    },
)

# Use one logical processor per CPU core for inference when hyper threading is on
compiled_model_3 = core.compile_model(
    model=model,
    device_name=device_name,
    config={properties.hint.enable_hyper_threading(): False},
)
# ! [ov:intel_cpu:multi_threading:part0]

# ! [ov:intel_cpu:multi_threading:part1]
# Disable CPU thread pinning for inference when the system supports it
compiled_model_4 = core.compile_model(
    model=model,
    device_name=device_name,
    config={properties.hint.enable_cpu_pinning(): False},
)
# ! [ov:intel_cpu:multi_threading:part1]
assert compiled_model_1
assert compiled_model_2
assert compiled_model_3
assert compiled_model_4
