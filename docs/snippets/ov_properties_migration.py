# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import openvino as ov
import openvino.properties as props
import openvino.properties.hint as hints
import openvino.properties.device as device
import openvino.properties.streams as streams

from utils import get_model

def main():
    core = ov.Core()

    # ! [core_set_property]
    core.set_property(device_name="CPU", properties={props.enable_profiling: True})
    # ! [core_set_property]

    model = get_model()

    if "GPU" not in core.available_devices:
        return 0

    # ! [core_compile_model]
    compiled_model = core.compile_model(model=model, device_name="MULTI", config=
        {
            device.priorities: "GPU,CPU",
            hints.performance_mode: hints.PerformanceMode.THROUGHPUT,
            hints.inference_precision: ov.Type.f32
        })
    # ! [core_compile_model]

    # ! [compiled_model_set_property]
    # turn CPU off for multi-device execution
    compiled_model.set_property(properties={device.priorities: "GPU"})
    # ! [compiled_model_set_property]

    # ! [core_get_rw_property]
    num_streams = core.get_property("CPU", streams.num)
    # ! [core_get_rw_property]

    # ! [core_get_ro_property]
    full_device_name = core.get_property("CPU", device.full_name)
    # ! [core_get_ro_property]

    # ! [compiled_model_get_rw_property]
    perf_mode = compiled_model.get_property(hints.performance_mode)
    # ! [compiled_model_get_rw_property]

    # ! [compiled_model_get_ro_property]
    nireq = compiled_model.get_property(props.optimal_number_of_infer_requests)
    # ! [compiled_model_get_ro_property]

    import ngraph as ng
    import openvino.inference_engine as ie
    from utils import get_ngraph_model

    core = ie.IECore()
    #! [core_get_metric]
    full_device_name = core.get_metric("CPU", "FULL_DEVICE_NAME")
    #! [core_get_metric]

    #! [core_get_config]
    num_streams = core.get_config("CPU", "CPU_THROUGHPUT_STREAMS")
    #! [core_get_config]

    #! [core_set_config]
    core.set_config({"PERF_COUNT": "YES"}, "CPU")
    #! [core_set_config]

    net = get_ngraph_model()

    #! [core_load_network]
    exec_network = core.load_network(net, "MULTI", {"DEVICE_PRIORITIES": "CPU, GPU",
                                                    "PERFORMANCE_HINT": "THROUGHPUT",
                                                    "ENFORCE_BF16": "NO"})
    #! [core_load_network]

    #! [executable_network_set_config]
    # turn CPU off for multi-device execution
    exec_network.set_config({"DEVICE_PRIORITIES": "GPU"})
    #! [executable_network_set_config]

    #! [executable_network_get_metric]
    nireq = exec_network.get_metric("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
    #! [executable_network_get_metric]

    #! [executable_network_get_config]
    perf_hint = exec_network.get_config("PERFORMANCE_HINT")
    #! [executable_network_get_config]
