# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino as ov
import openvino.properties as properties
import openvino.properties.device as device
import openvino.properties.streams as streams
from utils import get_model

model = get_model()


def MULTI_0():
    #! [MULTI_0]
    core = ov.Core()

    # Option 1
    # Pre-configure MULTI globally with explicitly defined devices,
    # and compile the model on MULTI using the newly specified default device list.
    core.set_property(
        device_name="MULTI", properties={device.priorities: "GPU,CPU"}
    )
    compiled_model = core.compile_model(model=model, device_name="MULTI")

    # Option 2
    # Specify the devices to be used by MULTI explicitly at compilation.
    # The following lines are equivalent:
    compiled_model = core.compile_model(model=model, device_name="MULTI:GPU,CPU")
    compiled_model = core.compile_model(
        model=model,
        device_name="MULTI",
        config={device.priorities: "GPU,CPU"},
    )
    #! [MULTI_0]


def MULTI_1():
    #! [MULTI_1]
    core = ov.Core()

    core.set_property(
        device_name="MULTI", properties={device.priorities: "CPU,GPU"}
    )
    # Once the priority list is set, you can alter it on the fly:
    # reverse the order of priorities
    core.set_property(
        device_name="MULTI", properties={device.priorities: "GPU,CPU"}
    )

    # exclude some devices (in this case, CPU)
    core.set_property(
        device_name="MULTI", properties={device.priorities: "GPU"}
    )

    # bring back the excluded devices
    core.set_property(
        device_name="MULTI", properties={device.priorities: "GPU,CPU"}
    )

    # You cannot add new devices on the fly!
    # Attempting to do so will trigger the following exception:
    # [ ERROR ] [NOT_FOUND] You can only change device
    # priorities but not add new devices with the model's
    # ov::device::priorities. CPU device was not in the original device list!
    #! [MULTI_1]


# the following two pieces of code appear not to be used anywhere
# they should be considered for removal


def available_devices_1():
    #! [available_devices_1]
    all_devices = "MULTI:"
    core = ov.Core()

    all_devices += ",".join(core.available_devices)
    compiled_model = core.compile_model(model=model, device_name=all_devices)
    #! [available_devices_1]


def available_devices_2():
    #! [available_devices_2]
    match_list = []
    all_devices = "MULTI:"
    dev_match_str = "GPU"
    core = ov.Core()

    for d in core.available_devices:
        if dev_match_str in d:
            match_list.append(d)
    all_devices += ",".join(match_list)
    compiled_model = core.compile_model(model=model, device_name=all_devices)
    #! [available_devices_2]


def MULTI_4():
    #! [MULTI_4]
    core = ov.Core()
    cpu_config = {streams.num : 4}
    gpu_config = {streams.num : 8}

    # When compiling the model on MULTI, configure CPU and GPU
    # (devices, priorities, and device configurations; gpu_config and cpu_config will load during compile_model() ):
    compiled_model = core.compile_model(
        model=model,
        device_name="MULTI:GPU,CPU",
        config={
            device.properties: {'CPU': cpu_config, 'GPU': gpu_config}
        }
    )

    # Optionally, query the optimal number of requests:
    nireq = compiled_model.get_property(
        properties.optimal_number_of_infer_requests
    )
    #! [MULTI_4]


def main():
    core = ov.Core()
    if "GPU" not in core.available_devices:
        return 0
    MULTI_0()
    MULTI_1()
    available_devices_1()
    available_devices_2()
    MULTI_4()
