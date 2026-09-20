# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""vLLM preset: expand options={"vllm": True} into per-flag defaults."""

import os
from typing import Optional


def set_pre_import_env() -> None:
    """Pin the env vLLM latches at import time. Must run before `import vllm`.

    VLLM_USE_LAYERNAME=1 hoists layer_name as an opaque graph input instead of
    a constant, which torchdynamo/compile.py can't handle. vllm.utils reads
    this at import, so setting it later or unpatching it after has no effect.

    Callers must invoke this explicitly (see setup docs) -- it is not run on
    import here, since this module is also reached by the generic (non-vLLM)
    torchdynamo path, which must not have this env var set for it.
    """
    os.environ["VLLM_USE_LAYERNAME"] = "0"


def bool_opt(options, key: str, default: bool) -> bool:
    """Resolve a boolean option: options[key] > vLLM preset > default.

    Strings "false"/"0" count as False. Generic torchdynamo callers should
    inline ``bool(options and options.get(key, default))`` instead — they do
    not need the preset lookup.
    """
    if options is not None and key in options:
        value = options[key]
    else:
        if is_vllm_preset(options) and has_preset_flag(key):
            value = preset_flag(key)
        else:
            return default
    return bool(value) and str(value).lower() not in ("false", "0")


# Expanded from options["vllm"]=True; caller-supplied flags win (see bool_opt).
_PRESET_FLAGS = {
    "unbind_affinity": True,
    "paged_attention": True,
    "pa_translate": True,
    "no_fallback": True,
    "fc_decompress": True,
    "dynamic_shapes": False,
}

# Model-independent OV CPU config defaults; caller-supplied keys win.
# Float-precision keys are absent -- derived per-model by precision_config().
_PRESET_CONFIG = {
    "DYNAMIC_QUANTIZATION_GROUP_SIZE": 32,
}

# Narrow float dtypes recognizable on a converted model. Anything else (f32, or
# a graph we could not read a dtype off) runs at _FALLBACK_FLOAT_PRECISION.
SUPPORTED_FLOAT_PRECISIONS = ("bf16", "f16")

# bf16 rather than f32: the narrow-float GEMM path is substantially faster, and
# f32 models are known to work under it.
_FALLBACK_FLOAT_PRECISION = "bf16"

# Recognized dtypes we deliberately do not compute in -> what we use instead.
# See precision_config() for why f16 is here.
_COMPUTE_PRECISION_SUBSTITUTE = {"f16": "bf16"}


def precision_config(model_precision: Optional[str] = None) -> dict:
    """Return the matching OV INFERENCE_PRECISION_HINT/KV_CACHE_PRECISION pair.

    Both keys must name the same type (OV CPU PagedAttention only instantiates
    matching compute/cache-type triples). f16 computes as bf16: vLLM's unfused
    RMSNorm overflows f16's range but not bf16's.
    """
    et = (model_precision if model_precision in SUPPORTED_FLOAT_PRECISIONS
          else _FALLBACK_FLOAT_PRECISION)
    et = _COMPUTE_PRECISION_SUBSTITUTE.get(et, et)
    return {"INFERENCE_PRECISION_HINT": et, "KV_CACHE_PRECISION": et}


def is_vllm_preset(options) -> bool:
    """True iff options["vllm"] is set to a truthy value."""
    if options is None or "vllm" not in options:
        return False
    value = options["vllm"]
    return bool(value) and str(value).lower() not in ("false", "0")


def preset_flag(key: str):
    """Return the preset value for `key`, or None if `key` is not in the preset."""
    return _PRESET_FLAGS.get(key)


def has_preset_flag(key: str) -> bool:
    return key in _PRESET_FLAGS


def merge_preset_config(base: Optional[dict], model_precision: Optional[str] = None) -> dict:
    """Fill in the preset OV-config defaults; entries in `base` take priority.

    `model_precision` is the model's float dtype, used to derive the precision
    pair — see precision_config().
    """
    out = dict(base or {})
    for key, value in _PRESET_CONFIG.items():
        out.setdefault(key, value)
    for key, value in precision_config(model_precision).items():
        out.setdefault(key, value)
    return out
