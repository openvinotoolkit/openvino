"""
 Copyright (C) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import multiprocessing
from .logging import logger

VPU_DEVICE_NAME = "VPU"
MYRIAD_DEVICE_NAME = "MYRIAD"
HDDL_DEVICE_NAME = "HDDL"
FPGA_DEVICE_NAME = "FPGA"
CPU_DEVICE_NAME = "CPU"
GPU_DEVICE_NAME = "GPU"
HETERO_DEVICE_NAME = "HETERO"
UNKNOWN_DEVICE_TYPE = "UNKNOWN"

DEVICE_DURATION_IN_SECS = {
    CPU_DEVICE_NAME: 60,
    GPU_DEVICE_NAME: 60,
    VPU_DEVICE_NAME: 60,
    MYRIAD_DEVICE_NAME: 60,
    HDDL_DEVICE_NAME: 60,
    FPGA_DEVICE_NAME: 120,
    UNKNOWN_DEVICE_TYPE: 120
}

DEVICE_NIREQ_ASYNC = {
    CPU_DEVICE_NAME: 2,
    GPU_DEVICE_NAME: 2,
    VPU_DEVICE_NAME: 4,
    MYRIAD_DEVICE_NAME: 4,
    HDDL_DEVICE_NAME: 100,
    FPGA_DEVICE_NAME: 3,
    UNKNOWN_DEVICE_TYPE: 1
}

def get_duration_in_secs(target_device):
    duration = 0
    for device in DEVICE_DURATION_IN_SECS:
        if device in target_device:
            duration = max(duration, DEVICE_DURATION_IN_SECS[device])

    if duration == 0:
        duration = DEVICE_DURATION_IN_SECS[UNKNOWN_DEVICE_TYPE]
        logger.warn("Default duration {} seconds is used for unknown device {}".format(duration, target_device))

    return duration

def get_nireq(target_device):
    nireq = 0
    for device in DEVICE_NIREQ_ASYNC:
        if device in target_device:
            nireq = max(nireq, DEVICE_NIREQ_ASYNC[device])

    if nireq == 0:
        nireq = DEVICE_NIREQ_ASYNC[UNKNOWN_DEVICE_TYPE]
        logger.warn("Default number of requests {} is used for unknown device {}".format(duration, target_device))

    return nireq

def parseDevices(device_string):
    devices = device_string
    if ':' in devices:
        devices = devices.partition(':')[2]
    return [ d[:d.index('(')] if '(' in d else d for d in devices.split(',') ]

def parseValuePerDevice(devices, values_string):
    ## Format: <device1>:<value1>,<device2>:<value2> or just <value>
    result = {}
    if not values_string:
      return result
    device_value_strings = values_string.upper().split(',')
    for device_value_string in device_value_strings:
        device_value_vec = device_value_string.split(':')
        if len(device_value_vec) == 2:
            for device in devices:
                if device == device_value_vec[0]:
                    value = int(device_value_vec[1])
                    result[device_value_vec[0]] = value
                    break
        elif len(device_value_vec) == 1:
            value = int(device_value_vec[0])
            for device in devices:
                result[device] = value
        elif not device_value_vec:
            raise Exception("Unknown string format: " + values_string)
    return result
