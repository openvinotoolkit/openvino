# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

from typing import Optional, Any
import os
from openvino.runtime import Core


def _get_device(options) -> Optional[Any]:
    core = Core()
    device = "CPU"

    if options is not None and "device" in options:
        device = options["device"]
    else:
        device = os.getenv("OPENVINO_TORCH_BACKEND_DEVICE")

    if device is not None:
        assert device in core.available_devices, (
            "Specified device "
            + device
            + " is not in the list of OpenVINO Available Devices"
        )
    else:
        device = "CPU"
    return device

def _is_cache_dir_in_config(options) -> Optional[Any]:
    if options is not None and "config" in options:
        cfg = options["config"]
        if cfg is not None and "CACHE_DIR" in cfg:
            return True
    return False


def _get_cache_dir(options) -> Optional[Any]:
    cache_dir = "./cache"
    if options is not None and "cache_dir" in options:
        cache_dir = options["cache_dir"]
    if _is_cache_dir_in_config(options):
        cache_dir = options["config"]["CACHE_DIR"]
    else:
        cache_dir_env = os.getenv("OPENVINO_TORCH_CACHE_DIR")
        if cache_dir_env is not None:
            cache_dir = cache_dir_env
    return cache_dir


def _get_model_caching(options) -> Optional[Any]:
    if options is not None and "model_caching" in options:
        caching = options["model_caching"]
        if bool(caching) and str(caching).lower() not in ["false", "0"]:
            return True
        else:
            return False
    else:
        caching = os.getenv("OPENVINO_TORCH_MODEL_CACHING")
        if caching is not None and caching.lower() not in ["false", "0"]:
            return True
        else:
            return False


def _get_config(options) -> Optional[Any]:
    if options is not None and "config" in options:
        return options["config"]
    return {}