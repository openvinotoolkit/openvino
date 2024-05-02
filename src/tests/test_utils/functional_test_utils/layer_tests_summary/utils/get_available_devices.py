# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from utils.conformance_utils import get_logger
from utils.constants import (
    DEVICE_ARCHITECTURE_PROPERTY,
    FULL_DEVICE_PROPERTY,
    SUPPORTED_PROPERTIES,
)

logger = get_logger("get_available_device")

try:
    from openvino.runtime import Core
except:
    import os

    from utils.conformance_utils import set_env_variable
    from utils.constants import LD_LIB_PATH_NAME, PY_OPENVINO
    from utils.file_utils import find_latest_dir, get_ov_path

    script_dir, _ = os.path.split(os.path.abspath(__file__))
    ov_bin_path = get_ov_path(script_dir, None, True)
    if PY_OPENVINO in os.listdir(ov_bin_path):
        env = os.environ
        py_ov = os.path.join(ov_bin_path, PY_OPENVINO)
        py_ov = os.path.join(py_ov, find_latest_dir(py_ov))

        env = set_env_variable(env, "PYTHONPATH", py_ov)
        env = set_env_variable(env, LD_LIB_PATH_NAME, ov_bin_path)
        logger.warning(
            "Set the following env varibles to rename conformance ir based on hash: "
        )
        logger.warning(f'PYTHONPATH={env["PYTHONPATH"]}')
        logger.warning(f"{LD_LIB_PATH_NAME}={env[LD_LIB_PATH_NAME]}")
        exit(0)
    else:
        logger.error(f"Impossible to run the tool! PyOpenVINO was not built!")
        exit(-1)


def get_available_devices(target_device=None, exclude_device=None):
    result = list()
    core = Core()
    if exclude_device is None:
        exclude_device = "NOT_EXISTED_DEVICE"
    for device in core.available_devices:
        if target_device is None or target_device in device:
            if exclude_device in device:
                continue
            supported_metrics = core.get_property(target_device, SUPPORTED_PROPERTIES)
            if FULL_DEVICE_PROPERTY in supported_metrics:
                if core.get_property(
                    target_device, FULL_DEVICE_PROPERTY
                ) != core.get_property(device, FULL_DEVICE_PROPERTY):
                    logger.warning(
                        f"Device {device} is different {FULL_DEVICE_PROPERTY} with {target_device} ( : {core.get_property(device, FULL_DEVICE_PROPERTY)} : {core.get_property(target_device, FULL_DEVICE_PROPERTY)} )"
                    )
                    continue
            if DEVICE_ARCHITECTURE_PROPERTY in supported_metrics:
                if not core.get_property(
                    target_device, DEVICE_ARCHITECTURE_PROPERTY
                ) != core.get_property(device, DEVICE_ARCHITECTURE_PROPERTY):
                    logger.warning(
                        f"Device {device} is different {DEVICE_ARCHITECTURE_PROPERTY} with {target_device} ( : {core.get_property(device, DEVICE_ARCHITECTURE_PROPERTY)} : {core.get_property(target_device, DEVICE_ARCHITECTURE_PROPERTY)} )"
                    )
                    continue
            logger.info(f"{device} is added to device pool")
            result.append(device)
    return result
