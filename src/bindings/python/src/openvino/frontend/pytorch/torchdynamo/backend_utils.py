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


# Caller can opt into a preset of defaults (currently the only one is the
# vLLM preset, lives in torchdynamo.vllm.preset). The generic backend_utils
# does not know about specific presets; it just delegates.
def _bool_opt(options, key: str, default: bool) -> bool:
    """Resolve a boolean plugin option.

    Priority: options[key] > preset (if active) > default.
    Strings "false"/"0" are treated as False.
    """
    if options is not None and key in options:
        v = options[key]
    else:
        # Check vLLM preset, if active. Imported lazily so backend_utils
        # stays usable without the vllm subpackage on disk.
        try:
            from openvino.frontend.pytorch.torchdynamo.vllm import preset as _preset
        except Exception:
            _preset = None
        if _preset is not None and _preset.is_vllm_preset(options) and _preset.has_preset_flag(key):
            v = _preset.preset_flag(key)
        else:
            return default
    return bool(v) and str(v).lower() not in ("false", "0")


def _config_with_vllm_defaults(options):
    """Return options["config"] (or a fresh dict), merged with the vLLM preset
    OV-config defaults when options["vllm"] is set. Caller-supplied config
    keys take priority."""
    base = dict(_get_config(options) or {})
    try:
        from openvino.frontend.pytorch.torchdynamo.vllm import preset as _preset
    except Exception:
        return base
    if _preset.is_vllm_preset(options):
        return _preset.merge_preset_config(base)
    return base
