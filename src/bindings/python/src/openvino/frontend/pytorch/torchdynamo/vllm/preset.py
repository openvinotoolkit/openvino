# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""vLLM preset: expand options={"vllm": True} into per-flag defaults."""

import os
from typing import Optional


def set_pre_import_env() -> None:
    """Pin the env vLLM latches at import time. Must run before `import vllm`.

    Only VLLM_USE_LAYERNAME, and it is forced rather than defaulted: the OV
    backend has no working configuration at 1. vllm.utils.torch_utils computes
    `_USE_LAYERNAME` and the LayerNameType op-schema alias at module import,
    and at 1 torch hoists layer_name as an opaque graph *input* instead of a
    constant -- torchdynamo/compile.py then calls .type() on it and dies.
    Plain str keeps layer_name the constant the PA translator and side_channel
    read off the FX node.

    Setting it any later is useless: vllm.utils.torch_utils imports ahead of
    vllm.platforms and vllm.plugins, so every entry-point group -- including
    the one plugin.register() uses -- loads after the value is frozen. Nor can
    it be undone from Python afterwards: the op schemas declare layer_name as
    `PyObject` and keep it even if _USE_LAYERNAME/LayerNameType are patched.
    """
    os.environ["VLLM_USE_LAYERNAME"] = "0"


# Run on import as well as on call, so importing anything from this package
# before vLLM is enough -- no caller has to remember. A no-op when the plugin
# entry point imports us, which is already past the point of no return.
set_pre_import_env()


def bool_opt(options, key: str, default: bool) -> bool:
    """Resolve a boolean option: options[key] > vLLM preset > default.

    Strings "false"/"0" count as False. Generic torchdynamo callers should
    inline ``bool(options and options.get(key, default))`` instead — they do
    not need the preset lookup.
    """
    if options is not None and key in options:
        v = options[key]
    else:
        if is_vllm_preset(options) and has_preset_flag(key):
            v = preset_flag(key)
        else:
            return default
    return bool(v) and str(v).lower() not in ("false", "0")


# Expanded from options["vllm"]=True; caller-supplied flags win (see bool_opt).
_PRESET_FLAGS = {
    "unbind_affinity": True,
    "paged_attention": True,
    "pa_translate": True,
    "no_fallback": True,
    "fc_decompress": True,
    "dynamic_shapes": False,
}

# Model-independent OV CPU config defaults; caller-supplied keys win. The
# float-precision keys are deliberately absent: they follow the model dtype and
# are derived per-model by precision_config().
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
    """Return the OV float-precision config pair for a model of the given dtype.

    `model_precision` is "bf16", "f16", or None/other for unknown. Both keys
    are derived from one value because they must name the *same*
    type: OV CPU PagedAttention selects
    ``AttentionExecutor<compute_t, key_cache_t, value_cache_t>`` and only
    instantiates <bf16,bf16,bf16>, <f16,f16,f16>, <f32,f16,f16>, <f32,f32,f32>
    (executor_pa.cpp). bf16 compute over an f16 cache is not among them, and
    compile_model throws ``expect kvcache type bf16, current: f16``.

    f16 models compute in bf16. vLLM builds RMSNorm from primitive ops
    (``custom_ops: ["none"]``), RMSFusion does not match that shape, so the
    reduction runs as Power -> ReduceMean at the inference precision; ``x**2``
    on real activations exceeds f16's 65504 ceiling and the norm returns NaN
    from the third token on. bf16 has f32's exponent range and cannot overflow
    there. Once the reduction is kept out of f16 -- RMSFusion firing, or
    marking it precision-sensitive -- dropping "f16" from
    _COMPUTE_PRECISION_SUBSTITUTE is the whole change.

    Computing at a precision other than the model dtype only works because
    compile_hooks.retype_kv_cache_parameters redeclares the cache Parameters to
    match; the frontend creates them at the model dtype, and that mismatch is
    what made the old hardcoded-bf16 preset reject every f16 model.
    """
    et = (model_precision if model_precision in SUPPORTED_FLOAT_PRECISIONS
          else _FALLBACK_FLOAT_PRECISION)
    et = _COMPUTE_PRECISION_SUBSTITUTE.get(et, et)
    return {"INFERENCE_PRECISION_HINT": et, "KV_CACHE_PRECISION": et}


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


def merge_preset_config(base: Optional[dict], model_precision: Optional[str] = None) -> dict:
    """Fill in the preset OV-config defaults; entries in `base` take priority.

    `model_precision` is the model's float dtype, used to derive the precision
    pair — see precision_config().
    """
    out = dict(base or {})
    for k, v in _PRESET_CONFIG.items():
        out.setdefault(k, v)
    for k, v in precision_config(model_precision).items():
        out.setdefault(k, v)
    return out
