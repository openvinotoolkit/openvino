# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import openvino.runtime as ov

device_name = 'CPU'
xml_path = 'model.xml'
core = ov.Core()
core.set_property("CPU", ov.properties.intel_cpu.sparse_weights_decompression_rate(0.8))
model = core.read_model(model=xml_path)
# ! [ov:intel_cpu:multi_threading_0:part0]
# Use one processor for inference
compiled_model = core.compile_model(model=model, device_name=device_name, config={"INFERENCE_NUM_THREADS": "1"})

# Use processors of Efficient-cores for inference on hybrid platform
compiled_model = core.compile_model(model=model, device_name=device_name, config={"SCHEDULING_CORE_TYPE": "ECORE_ONLY"})

# Use one processor per core for inference when hyper threading is on
compiled_model = core.compile_model(model=model, device_name=device_name, config={"ENABLE_HYPER_THREADING": "False"})
# ! [ov:intel_cpu:multi_threading_0:part0]

# ! [ov:intel_cpu:multi_threading_0:part1]
# Disable CPU threads pinning for inference when system supoprt it
compiled_model = core.compile_model(model=model, device_name=device_name, config={"ENABLE_CPU_PINNING": "False"})
# ! [ov:intel_cpu:multi_threading_0:part1]
assert compiled_model
