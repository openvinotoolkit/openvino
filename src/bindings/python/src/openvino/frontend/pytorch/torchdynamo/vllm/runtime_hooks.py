# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# mypy: ignore-errors

"""vLLM PagedAttention binding for torchdynamo.execute's infer path."""

import os

from .side_channel import _bind_gdn_side_channel, _bind_paged_attention_side_channel


# Per-InferRequest caches for the OV_FAST_INFER fast path.
_fastinfer_port_cache = {}
_fastinfer_bound_ids = {}   # id(req) -> [[val_id, ov_tensor_ref], ...] per port
_fastinfer_out_cache = {}   # id(req) -> {out_port: numpy_view}
_fastinfer_out_static = {}  # id(compiled) -> bool: output views are reusable


def clear_caches():
    """Drop the fast-infer caches: their id() keys are reused once the objects are freed."""
    for cache in (_fastinfer_port_cache, _fastinfer_bound_ids, _fastinfer_out_cache, _fastinfer_out_static):
        cache.clear()


# Returned by run_pa_infer: skip this infer and run eager.
class _PaSkip:
    __slots__ = ()


PA_SKIP = _PaSkip()


def run_pa_infer(compiled, req, ov_inputs):
    """Infer with the PA side channel bound.

    Returns PA_SKIP during warmup (run eager), {output port: numpy view}, or None without ``__pa__`` inputs.
    """
    if not has_pa_inputs(compiled):
        return None
    if should_skip_pa_infer():
        return PA_SKIP
    call_kwargs, scatter_back = build_call_kwargs(compiled, ov_inputs)
    if not call_kwargs:
        return None
    result = infer_with_pa(req, compiled, call_kwargs)
    if scatter_back is not None:
        scatter_back()
    return result


def has_pa_inputs(compiled) -> bool:
    """Return True if any input of `compiled` is a ``__pa__`` Parameter."""
    for inp in compiled.inputs:
        names = inp.get_names()
        if not names:
            continue
        for name in names:
            if name.startswith("__pa__"):
                return True
    return False


def should_skip_pa_infer() -> bool:
    """Detect vLLM warmup / profile_run (empty attn_metadata), whose output nobody reads, so eager is safe.

    A missing forward context returns False on purpose: eager can't write the KV cache, so skipping a real step would
    decode garbage silently, while PA without a context fails loudly in the binder.
    """
    try:
        from vllm.forward_context import get_forward_context
    except Exception:
        return False
    try:
        ctx = get_forward_context()
    except (AssertionError, RuntimeError):
        return False
    if ctx is None:
        return False
    am = getattr(ctx, "attn_metadata", None)
    # None at bootstrap, an empty dict during profile_run.
    if am is None:
        return True
    if isinstance(am, dict) and not am:
        return True
    return False


def build_call_kwargs(compiled, ov_inputs):
    """Map ``__pa__`` inputs to side-channel tensors and the rest to `ov_inputs` in order.

    Returns (call_kwargs, scatter_back): call_kwargs is None without PA inputs; scatter_back must run after infer.
    """
    pa_inputs_by_pos = _bind_paged_attention_side_channel(compiled)
    gdn_inputs, scatter_back = _bind_gdn_side_channel(compiled)
    pa_inputs_by_pos.update(gdn_inputs)
    if not pa_inputs_by_pos:
        return None, None
    call_kwargs = {}
    positional = []
    for inp in compiled.inputs:
        pa_tensor = None
        for name in inp.get_names():
            if name.startswith("__pa__") and name in pa_inputs_by_pos:
                pa_tensor = pa_inputs_by_pos[name]
                break
        if pa_tensor is not None:
            call_kwargs[inp] = pa_tensor
        else:
            positional.append(inp)
    # Values bind by position; a count mismatch means a dropped or extra input would shift every later port.
    if len(positional) != len(ov_inputs):
        raise RuntimeError(f"PA graph has {len(positional)} positional inputs but got {len(ov_inputs)} values")
    call_kwargs.update(zip(positional, ov_inputs))
    return call_kwargs, scatter_back


def infer_with_pa(req, compiled, call_kwargs):
    """Run req.infer with `call_kwargs`; OV_FAST_INFER=1 skips rebinding unchanged inputs.

    Falls back to plain req.infer only if binding fails. Inference errors propagate: inference already advanced the KV
    cache and GDN state, so a rerun would advance them twice.
    """
    if os.environ.get("OV_FAST_INFER", "0") == "0":
        return req.infer(call_kwargs, share_inputs=True, share_outputs=True)

    import numpy as np
    import openvino as _ov
    try:
        _pc_key = id(compiled)
        _ports = _fastinfer_port_cache.get(_pc_key)
        if _ports is None:
            _ports = list(compiled.inputs)
            _fastinfer_port_cache[_pc_key] = _ports
        _req_key = id(req)
        _bound = _fastinfer_bound_ids.get(_req_key)
        if _bound is None or len(_bound) != len(_ports):
            _bound = [[0, None] for _ in range(len(_ports))]
            _fastinfer_bound_ids[_req_key] = _bound
        for _pi_idx, _port in enumerate(_ports):
            _val = call_kwargs.get(_port)
            if _val is None:
                raise RuntimeError("call_kwargs missing port")
            _val_id = id(_val)
            _slot = _bound[_pi_idx]
            if _val_id == _slot[0] and _slot[1] is not None:
                continue  # already bound; ov.Tensor wrapper kept alive
            if isinstance(_val, _ov.Tensor):
                _t = _val
            elif _port.get_element_type() == _ov.Type.bf16 and getattr(_val, "dtype", None) == np.float16:
                # bf16 arrives as float16 bits; wrap them with the port's type, or set_tensor rejects f16.
                _t = _ov.Tensor(_val, _val.shape, _ov.Type.bf16)
            else:
                _t = _ov.Tensor(_val, shared_memory=True)
            req.set_tensor(_port, _t)
            _slot[0] = _val_id
            _slot[1] = _t  # keep alive
    except Exception:
        _fastinfer_bound_ids.pop(id(req), None)
        _fastinfer_out_cache.pop(id(req), None)
        return req.infer(call_kwargs, share_inputs=True, share_outputs=True)
    try:
        req.infer(share_outputs=True)
        # Cached output views pin the first call's shape, so reuse them only for static outputs.
        _static_out = _fastinfer_out_static.get(_pc_key)
        if _static_out is None:
            _static_out = all(o.get_partial_shape().is_static for o in compiled.outputs)
            _fastinfer_out_static[_pc_key] = _static_out
        if not _static_out:
            return {out: req.get_tensor(out).data for out in compiled.outputs}
        _out = _fastinfer_out_cache.get(_req_key)
        if _out is None:
            _out = {out: req.get_tensor(out).data for out in compiled.outputs}
            _fastinfer_out_cache[_req_key] = _out
        return _out
    except Exception:
        _fastinfer_bound_ids.pop(id(req), None)
        _fastinfer_out_cache.pop(id(req), None)
        raise
