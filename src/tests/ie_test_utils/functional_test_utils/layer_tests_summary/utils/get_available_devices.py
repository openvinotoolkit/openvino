# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.runtime import Core

def get_available_devices(target_device = None, exclude_device = None):
    result = list()
    core = Core()
    if exclude_device is None:
        exclude_device = "NOT_EXISTED_DEVICE"
    for device in core.available_devices:
        if target_device is None or target_device in device:
            if exclude_device in device:
                continue
            result.append(device)
    return result
