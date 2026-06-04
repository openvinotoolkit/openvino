# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# mypy: ignore-errors

from typing import Optional, Any
from openvino import Core


def _get_device(options) -> Optional[Any]:
    core = Core()
    device = "CPU"

    if options is not None and "device" in options:
        device = options["device"]

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
    return cache_dir


def _get_aot_autograd(options) -> Optional[Any]:
    if options is not None and "aot_autograd" in options:
        aot_autograd = options["aot_autograd"]
        if bool(aot_autograd) and str(aot_autograd).lower() not in ["false", "0"]:
            return True
        else:
            return False


def _get_model_caching(options) -> Optional[Any]:
    if options is not None and "model_caching" in options:
        caching = options["model_caching"]
        if bool(caching) and str(caching).lower() not in ["false", "0"]:
            return True
    return False


def _get_config(options) -> Optional[Any]:
    if options is not None and "config" in options:
        return options["config"]
    return {}


def _get_decompositions(options) -> Optional[Any]:
    decompositions = []
    if options is not None and "decompositions" in options:
        decompositions = options["decompositions"]
    return decompositions


def _get_disabled_ops(options) -> Optional[Any]:
    disabled_ops = []
    if options is not None and "disabled_ops" in options:
        disabled_ops = options["disabled_ops"]
    return disabled_ops


def _is_testing(options) -> Optional[Any]:
    if options is not None and "testing" in options:
        is_testing = options["testing"]
        if bool(is_testing) and str(is_testing).lower not in ["false", "0"]:
            return True
    return False


# Mega-preset: when options["vllm"] is truthy, expand into individual flags
# unless the user already set them explicitly. Keeps caller code one-liner.
_VLLM_PRESET_FLAGS = {
    "unbind_affinity": True,
    "paged_attention": True,
    "pa_translate": True,
    "no_fallback": True,
    "fc_decompress": True,
    "dynamic_shapes": False,
}
_VLLM_PRESET_CONFIG = {
    "KV_CACHE_PRECISION": "bf16",
    "INFERENCE_PRECISION_HINT": "bf16",
    "DYNAMIC_QUANTIZATION_GROUP_SIZE": 32,
}


def _is_vllm_preset(options) -> bool:
    if options is None or "vllm" not in options:
        return False
    v = options["vllm"]
    return bool(v) and str(v).lower() not in ("false", "0")


def _bool_opt(options, key: str, env: str, default: bool) -> bool:
    """Resolve a boolean plugin option.

    Priority: options[key] > vLLM preset (if active) > env > default.
    Strings "false"/"0" are treated as False.
    """
    import os as _os_b
    if options is not None and key in options:
        v = options[key]
    elif _is_vllm_preset(options) and key in _VLLM_PRESET_FLAGS:
        v = _VLLM_PRESET_FLAGS[key]
    elif env in _os_b.environ:
        v = _os_b.environ[env]
    else:
        return default
    return bool(v) and str(v).lower() not in ("false", "0")


def _config_with_vllm_defaults(options):
    """Return options["config"] (or a fresh dict), merged with vLLM preset
    defaults when options["vllm"] is set. Caller-supplied config keys win."""
    base = dict(_get_config(options) or {})
    if _is_vllm_preset(options):
        for k, v in _VLLM_PRESET_CONFIG.items():
            base.setdefault(k, v)
    return base
