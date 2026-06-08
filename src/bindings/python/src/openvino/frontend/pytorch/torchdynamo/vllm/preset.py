# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""vLLM preset: expand options={"vllm": True} into per-flag defaults.

Used by torchdynamo.backend_utils._bool_opt and torchdynamo.compile to apply
vLLM-specific defaults when the caller opts into the preset. Lives in the
vllm/ subpackage so that the generic torchdynamo backend stays free of
vLLM-specific knowledge.
"""

from typing import Optional, Any

# Per-flag defaults expanded from options["vllm"]=True. Caller-supplied flags
# take priority over these (see _bool_opt).
_PRESET_FLAGS = {
    "unbind_affinity": True,
    "paged_attention": True,
    "pa_translate": True,
    "no_fallback": True,
    "fc_decompress": True,
    "dynamic_shapes": False,
}

# OV CPU-config defaults expanded from options["vllm"]=True. Caller-supplied
# config keys win.
_PRESET_CONFIG = {
    "KV_CACHE_PRECISION": "bf16",
    "INFERENCE_PRECISION_HINT": "bf16",
    "DYNAMIC_QUANTIZATION_GROUP_SIZE": 32,
}


def is_vllm_preset(options) -> bool:
    """True iff options["vllm"] is set to a truthy value."""
    if options is None or "vllm" not in options:
        return False
    v = options["vllm"]
    return bool(v) and str(v).lower() not in ("false", "0")


def preset_flag(key: str):
    """Return the preset value for `key`, or None if `key` is not in the preset."""
    return _PRESET_FLAGS.get(key)


def has_preset_flag(key: str) -> bool:
    return key in _PRESET_FLAGS


def merge_preset_config(base: Optional[dict]) -> dict:
    """Return a dict with the preset OV-config defaults filled in. Caller-supplied
    entries in `base` take priority."""
    out = dict(base or {})
    for k, v in _PRESET_CONFIG.items():
        out.setdefault(k, v)
    return out
