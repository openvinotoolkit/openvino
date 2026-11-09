# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# mypy: ignore-errors

"""vLLM preset: expand options={"vllm": True} into per-flag defaults."""

import os
from typing import Optional


def set_pre_import_env() -> None:
    """Set VLLM_USE_LAYERNAME=0 (an opaque layer_name input breaks compile.py); must run before `import vllm`.

    Not run on import: the generic torchdynamo path also imports this module.
    """
    os.environ["VLLM_USE_LAYERNAME"] = "0"


def bool_opt(options, key: str, default: bool) -> bool:
    """Resolve a boolean option: options[key] > vLLM preset > default; "false"/"0" count as False."""
    if options is not None and key in options:
        value = options[key]
    elif key in _PRESET_FLAGS and is_vllm_preset(options):
        value = _PRESET_FLAGS[key]
    else:
        return default
    return bool(value) and str(value).lower() not in ("false", "0")


# Expanded from options["vllm"]=True; caller-supplied flags win.
_PRESET_FLAGS = {
    "paged_attention": True,
    "pa_translate": True,
    "no_fallback": True,
    "canonical_precision": True,
    "dynamic_shapes": False,
}

# Model-independent OV CPU config defaults; caller-supplied keys win. Precision comes from precision_config().
_PRESET_CONFIG = {
    # On-the-fly int8 FC activations, as in OV GenAI on CPU.
    "DYNAMIC_QUANTIZATION_GROUP_SIZE": 32,
}

# Anything else (f32, or no readable dtype) runs at _FALLBACK_FLOAT_PRECISION.
SUPPORTED_FLOAT_PRECISIONS = ("bf16", "f16")

# bf16 rather than f32: much faster, and f32 models work under it.
_FALLBACK_FLOAT_PRECISION = "bf16"

# Recognized dtypes we do not compute in -> what we use instead.
_COMPUTE_PRECISION_SUBSTITUTE = {"f16": "bf16"}


def _amx_bf16_supported() -> bool:
    """Whether the CPU has AMX-BF16, which OV CPU PagedAttention needs for a bf16 KV cache (same check as vLLM)."""
    try:
        import torch
        return bool(torch.cpu._is_amx_tile_supported())
    except Exception:
        return False


def precision_config(model_precision: Optional[str] = None) -> dict:
    """Return matching INFERENCE_PRECISION_HINT/KV_CACHE_PRECISION values (CPU PagedAttention needs them equal).

    f16 computes as bf16 (vLLM's unfused RMSNorm overflows f16), and bf16 falls back to f32 without AMX-BF16.
    """
    et = (model_precision if model_precision in SUPPORTED_FLOAT_PRECISIONS
          else _FALLBACK_FLOAT_PRECISION)
    et = _COMPUTE_PRECISION_SUBSTITUTE.get(et, et)
    if et == "bf16" and not _amx_bf16_supported():
        et = "f32"
    return {"INFERENCE_PRECISION_HINT": et, "KV_CACHE_PRECISION": et}


def is_vllm_preset(options) -> bool:
    """True iff options["vllm"] is set to a truthy value."""
    return bool_opt(options, "vllm", False)
