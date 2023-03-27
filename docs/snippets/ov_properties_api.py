# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from openvino.runtime import Core

# [get_available_devices]
core = Core()
available_devices = core.available_devices
# [get_available_devices]

# [hetero_priorities]
device_priorites = core.get_property("HETERO", "MULTI_DEVICE_PRIORITIES")
# [hetero_priorities]

# [cpu_device_name]
cpu_device_name = core.get_property("CPU", "FULL_DEVICE_NAME")
# [cpu_device_name]

model = core.read_model(model="sample.xml")
# [compile_model_with_property]
config = {"PERFORMANCE_HINT": "THROUGHPUT",
        "INFERENCE_PRECISION_HINT": "f32"}
compiled_model = core.compile_model(model, "CPU", config)
# [compile_model_with_property]

# [optimal_number_of_infer_requests]
compiled_model = core.compile_model(model, "CPU")
nireq = compiled_model.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
# [optimal_number_of_infer_requests]


# [core_set_property_then_compile]
# latency hint is a default for CPU
core.set_property("CPU", {"PERFORMANCE_HINT": "LATENCY"})
# compiled with latency configuration hint
compiled_model_latency = core.compile_model(model, "CPU")
# compiled with overriden performance hint value
config = {"PERFORMANCE_HINT": "THROUGHPUT"}
compiled_model_thrp = core.compile_model(model, "CPU", config)
# [core_set_property_then_compile]


# [inference_num_threads]
compiled_model = core.compile_model(model, "CPU")
nthreads = compiled_model.get_property("INFERENCE_NUM_THREADS")
# [inference_num_threads]

# [multi_device]
config = {"MULTI_DEVICE_PRIORITIES": "CPU,GPU"}
compiled_model = core.compile_model(model, "MULTI", config)
# change the order of priorities
compiled_model.set_property({"MULTI_DEVICE_PRIORITIES": "GPU,CPU"})
# [multi_device]
