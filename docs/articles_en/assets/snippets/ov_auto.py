# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
 
#! [py_ov_property_import_header]
import openvino as ov
import openvino.properties as properties
import openvino.properties.device as device
import openvino.properties.hint as hints
import openvino.properties.streams as streams
import openvino.properties.intel_auto as intel_auto
#! [py_ov_property_import_header]
import openvino.properties.log as log

from utils import get_model

model = get_model()


def part0():
    #! [part0]
    core = ov.Core()

    #  compile a model on AUTO using the default list of device candidates.
    #  The following lines are equivalent:
    compiled_model = core.compile_model(model=model)
    compiled_model = core.compile_model(model=model, device_name="AUTO")

    # Optional
    # You can also specify the devices to be used by AUTO.
    # The following lines are equivalent:
    compiled_model = core.compile_model(model=model, device_name="AUTO:GPU,CPU")
    compiled_model = core.compile_model(
        model=model,
        device_name="AUTO",
        config={device.priorities: "GPU,CPU"},
    )

    # Optional
    # the AUTO plugin is pre-configured (globally) with the explicit option:
    core.set_property(
        device_name="AUTO", properties={device.priorities: "GPU,CPU"}
    )
    #! [part0]


def part3():
    #! [part3]
    core = ov.Core()

    # Compile a model on AUTO with Performance Hints enabled:
    # To use the “THROUGHPUT” mode:
    compiled_model = core.compile_model(
        model=model,
        device_name="AUTO",
        config={
            hints.performance_mode: hints.PerformanceMode.THROUGHPUT
        },
    )
    # To use the “LATENCY” mode:
    compiled_model = core.compile_model(
        model=model,
        device_name="AUTO",
        config={
            hints.performance_mode: hints.PerformanceMode.LATENCY
        },
    )
    # To use the “CUMULATIVE_THROUGHPUT” mode:
    # To use the ROUND_ROBIN schedule policy:
    compiled_model = core.compile_model(
        model=model,
        device_name="AUTO",
        config={
            hints.performance_mode: hints.PerformanceMode.CUMULATIVE_THROUGHPUT,
            intel_auto.schedule_policy: intel_auto.SchedulePolicy.ROUND_ROBIN
        },
    )
    #! [part3]


def part4():
    #! [part4]
    core = ov.Core()

    # Example 1
    compiled_model0 = core.compile_model(
        model=model,
        device_name="AUTO",
        config={hints.model_priority: hints.Priority.HIGH},
    )
    compiled_model1 = core.compile_model(
        model=model,
        device_name="AUTO",
        config={
            hints.model_priority: hints.Priority.MEDIUM
        },
    )
    compiled_model2 = core.compile_model(
        model=model,
        device_name="AUTO",
        config={hints.model_priority: hints.Priority.LOW},
    )
    # Assume that all the devices (CPU and GPUs) can support all the networks.
    # Result: compiled_model0 will use GPU.1, compiled_model1 will use GPU.0, compiled_model2 will use CPU.

    # Example 2
    compiled_model3 = core.compile_model(
        model=model,
        device_name="AUTO",
        config={hints.model_priority: hints.Priority.HIGH},
    )
    compiled_model4 = core.compile_model(
        model=model,
        device_name="AUTO",
        config={
            hints.model_priority: hints.Priority.MEDIUM
        },
    )
    compiled_model5 = core.compile_model(
        model=model,
        device_name="AUTO",
        config={hints.model_priority: hints.Priority.LOW},
    )
    # Assume that all the devices (CPU ang GPUs) can support all the networks.
    # Result: compiled_model3 will use GPU.1, compiled_model4 will use GPU.1, compiled_model5 will use GPU.0.
    #! [part4]


def part5():
    #! [part5]
    core = ov.Core()

    # gpu_config and cpu_config will load during compile_model()
    gpu_config = {
        hints.performance_mode: hints.PerformanceMode.THROUGHPUT,
        streams.num: 4
    }
    cpu_config = {
        hints.performance_mode: hints.PerformanceMode.LATENCY,
        streams.num: 8,
        properties.enable_profiling: True
    }
    compiled_model = core.compile_model(
        model=model,
        device_name="AUTO",
        config={
            device.priorities: "GPU,CPU",
            device.properties: {'CPU': cpu_config, 'GPU': gpu_config}
        }
    )
    #! [part5]


def part6():
    #! [part6]
    core = ov.Core()

    # compile a model on AUTO and set log level to debug
    compiled_model = core.compile_model(
        model=model,
        device_name="AUTO",
        config={log.level: log.Level.DEBUG},
    )
    # set log level with set_property and compile model
    core.set_property(
        device_name="AUTO",
        properties={log.level: log.Level.DEBUG},
    )
    compiled_model = core.compile_model(model=model, device_name="AUTO")
    #! [part6]


def part7():
    #! [part7]
    core = ov.Core()

    # compile a model on AUTO and set log level to debug
    compiled_model = core.compile_model(model=model, device_name="AUTO")
    # query the runtime target devices on which the inferences are being executed
    execution_devices = compiled_model.get_property(properties.execution_devices)
    #! [part7]


def main():
    part3()
    part4()
    part5()
    part6()
    part7()
    core = ov.Core()
    if "GPU" not in core.available_devices:
        return 0
    part0()
    part1()
