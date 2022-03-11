# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from openvino.runtime import Core


core = Core()

# ! [core_set_property]
core.set_property(device_name="CPU", properties={"PERF_COUNT": "YES"})
# ! [core_set_property]

model = core.read_model("sample.xml")

# ! [core_compile_model]
compiled_model = core.compile_model(model=model, device_name="MULTI", config=
    {
        "MULTI_DEVICE_PRIORITIES": "GPU,CPU",
        "PERFORMANCE_HINT": "THROUGHPUT",
        "INFERENCE_PRECISION_HINT": "f32"
    })
# ! [core_compile_model]

# ! [compiled_model_set_property]
# turn CPU off for multi-device execution
compiled_model.set_property(properties={"MULTI_DEVICE_PRIORITIES": "GPU"})
# ! [compiled_model_set_property]

# ! [core_get_rw_property]
num_streams = core.get_property("CPU", "NUM_STREAMS")
# ! [core_get_rw_property]

# ! [core_get_ro_property]
full_device_name = core.get_property("CPU", "FULL_DEVICE_NAME")
# ! [core_get_ro_property]

# ! [compiled_model_get_rw_property]
perf_model = compiled_model.get_property("PERFORMANCE_HINT")
# ! [compiled_model_get_rw_property]

# ! [compiled_model_get_ro_property]
nireq = compiled_model.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
# ! [compiled_model_get_ro_property]
