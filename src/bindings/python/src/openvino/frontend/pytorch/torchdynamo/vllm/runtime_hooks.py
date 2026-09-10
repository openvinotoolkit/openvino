# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""vLLM-specific runtime hooks for the OV inference path.

Called from torchdynamo.execute to keep the generic infer code free of
vLLM-specific PA-binding knowledge.
"""

import os

from .side_channel import _bind_paged_attention_side_channel


# Per-InferRequest caches for the OV_FAST_INFER fast path.
_fastinfer_port_cache = {}
_fastinfer_bound_ids = {}   # id(req) -> [[val_id, ov_tensor_ref], ...] per port
_fastinfer_out_cache = {}   # id(req) -> {out_port: numpy_view}
_fastinfer_out_static = {}  # id(compiled) -> bool: output views are reusable


# Sentinel returned by run_pa_infer to signal "skip this infer; use eager".
class _PA_Skip:
    __slots__ = ()


PA_SKIP = _PA_Skip()


def run_pa_infer(compiled, req, ov_inputs):
    """PA-side-channel infer entry point, called from execute.openvino_execute.

    Returns one of:
      * ``PA_SKIP`` — vLLM warmup/profile_run; run eager gm(*args) instead.
      * ``dict``    — infer result, OV output port -> numpy view. Caller wraps
                      with torch.from_numpy(...).
      * ``None``    — no ``__pa__`` inputs; use the normal positional
                      ``req.infer(ov_inputs, ...)`` path.
    """
    if not has_pa_inputs(compiled):
        return None
    if should_skip_pa_infer():
        return PA_SKIP
    call_kwargs = build_call_kwargs(compiled, ov_inputs)
    if not call_kwargs:
        return None
    return infer_with_pa(req, compiled, call_kwargs)


def has_pa_inputs(compiled) -> bool:
    """Return True if any compiled.inputs[] has a ``__pa__`` Parameter name."""
    for inp in compiled.inputs:
        names = inp.get_names()
        if not names:
            continue
        for n in names:
            if n.startswith("__pa__"):
                return True
    return False


def should_skip_pa_infer() -> bool:
    """Detect the vLLM warm-up / profile_run state.

    True when ForwardContext exists but ``attn_metadata`` is None. There the
    side-channel binder can only supply zero-length metadata, so the OV CPU
    PA kernel would read uninitialized ``_slot_mapping`` entries -- heap
    garbage, hence OOB writes.

    vLLM calls forward() in this state for determine_available_memory and for
    dummy_run; neither consumes the output semantically, so zeros of the
    expected shape substitute safely for a real infer.

    Any exception falls through to False, so real inference is never skipped
    by accident.
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
    # None at bootstrap, otherwise a dict keyed by layer. Empty during
    # profile_run, which likewise means "no real attention state".
    if am is None:
        return True
    if isinstance(am, dict) and not am:
        return True
    return False


def build_call_kwargs(compiled, ov_inputs):
    """Build the ``req.infer(...)`` kwargs dict for a PA-equipped graph.

    Walks compiled.inputs in order, mapping each ``__pa__`` Parameter to its
    bound side-channel tensor and every other one to the next entry of
    ``ov_inputs``. Returns None when there are no PA inputs, in which case the
    caller should pass ``ov_inputs`` directly.
    """
    pa_inputs_by_pos = _bind_paged_attention_side_channel(compiled)
    if not pa_inputs_by_pos:
        return None
    call_kwargs = {}
    tensor_pos = 0
    for inp in compiled.inputs:
        pa_tensor = None
        for n in inp.get_names():
            if n.startswith("__pa__") and n in pa_inputs_by_pos:
                pa_tensor = pa_inputs_by_pos[n]
                break
        if pa_tensor is not None:
            call_kwargs[inp] = pa_tensor
        else:
            call_kwargs[inp] = ov_inputs[tensor_pos]
            tensor_pos += 1
    return call_kwargs


def infer_with_pa(req, compiled, call_kwargs):
    """Run req.infer with the vLLM PA-side-channel call_kwargs.

    Under OV_FAST_INFER=1, a per-request cache skips ``set_tensor`` for ports
    whose value id is unchanged and reuses the output-view dict -- the latter
    only for statically-shaped outputs (see below). Falls back to the
    dict-based ``req.infer(call_kwargs, ...)`` on any error, which is also what
    runs when OV_FAST_INFER is unset, so execute.py's call site is identical
    either way.
    """
    if os.environ.get("OV_FAST_INFER", "0") == "0":
        return req.infer(call_kwargs, share_inputs=True, share_outputs=True)

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
            _t = _val if isinstance(_val, _ov.Tensor) else _ov.Tensor(_val, shared_memory=True)
            req.set_tensor(_port, _t)
            _slot[0] = _val_id
            _slot[1] = _t  # keep alive
        req.infer()
        # The cached views carry the shape and address the output buffers had
        # when the dict was built, which only holds while output shapes cannot
        # change. A dynamic-output model may re-allocate them every call, so
        # reusing the views hands back the *first* call's shape forever -- seen
        # as a 26-row hidden_states for a 6-token prefill once one compiled
        # model served every prefill length, which vLLM then indexes OOB.
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
        return req.infer(call_kwargs, share_inputs=True, share_outputs=True)
