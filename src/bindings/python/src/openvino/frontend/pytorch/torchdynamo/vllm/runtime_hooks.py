# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""vLLM-specific runtime hooks for the OV inference path.

Helpers called from torchdynamo.execute to keep the generic infer code
free of vLLM-specific PA-binding knowledge.
"""

import os

from .side_channel import _bind_paged_attention_side_channel


# Per-InferRequest caches for the OV_FAST_INFER fast path. Kept here so
# execute.py stays free of vLLM-specific state.
_fastinfer_port_cache = {}
_fastinfer_bound_ids = {}   # id(req) -> [[val_id, ov_tensor_ref], ...] per port
_fastinfer_out_cache = {}   # id(req) -> {out_port: numpy_view}
_fastinfer_out_static = {}  # id(compiled) -> bool: output views are reusable


# Sentinel returned by run_pa_infer to signal "skip this infer; use eager".
class _PA_Skip:
    __slots__ = ()
PA_SKIP = _PA_Skip()


def run_pa_infer(compiled, req, ov_inputs):
    """Consolidated PA-side-channel infer entry point called from
    torchdynamo.execute.openvino_execute.

    Returns one of:
      * ``PA_SKIP``       — vLLM warmup/profile_run state; caller should
                            run eager gm(*args) and skip real inference.
      * ``dict``          — the raw result of the infer call (mapping
                            OV output port -> numpy view). Caller should
                            wrap with torch.from_numpy(...).
      * ``None``          — this compiled model has no ``__pa__`` inputs;
                            caller should run its normal positional
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
    """Detect the vLLM warm-up / profile_run state where ForwardContext exists
    but ``attn_metadata`` is None. In that state the OV CPU PA kernel would
    read uninitialized ``_slot_mapping`` entries (heap garbage → OOB writes)
    because our side-channel binder can only supply zero-length metadata.

    vLLM invokes ``model.forward()`` in this state for two purposes:
      1. ``determine_available_memory`` — measuring peak activation memory.
      2. ``dummy_run`` — compile warm-up so torch.compile traces the graph.

    Neither consumes the model output semantically, so returning zeros of
    the expected shape is a safe substitute for a real infer call.

    Returns True only when the OV backend is active AND we can prove
    attn_metadata is missing. Any exception falls through to False so
    real inference is never skipped by accident.
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
    # attn_metadata is either None (bootstrap) or a dict keyed by layer.
    # An empty dict during profile_run also means "no real attention state".
    if am is None:
        return True
    if isinstance(am, dict) and not am:
        return True
    return False


def build_call_kwargs(compiled, ov_inputs):
    """Build the ``req.infer(...)`` kwargs dict for a PA-equipped graph.

    Walks compiled.inputs in order, mapping each ``__pa__``-named Parameter
    to its bound side-channel tensor and each remaining Parameter to the
    next entry of the user-supplied ``ov_inputs`` list. Returns None when
    no PA inputs are present, in which case the caller should pass
    ``ov_inputs`` directly.
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

    When OV_FAST_INFER=1 is set, this uses a per-request cache that skips
    ``set_tensor`` for ports whose value id has not changed since the last
    call, and reuses the output-view numpy dict across calls -- the latter only
    for models whose outputs are statically shaped, since a dynamic output is
    re-allocated per call and its cached view would report the first call's
    shape. Falls back to the dict-based ``req.infer(call_kwargs, ...)`` path on
    any error.

    When OV_FAST_INFER is unset (default), this is a thin wrapper around
    ``req.infer(call_kwargs, share_inputs=True, share_outputs=True)`` so the
    call site in execute.py stays identical for both paths.
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
        # The cached views are numpy arrays over the request's output buffers,
        # so they carry the shape and address those buffers had when the dict
        # was built. That only stays true while the output shapes cannot
        # change. A model with dynamic outputs re-infers (and may re-allocate)
        # its output tensors on every call, so reusing the views hands the
        # caller the *first* call's shape forever -- seen as a 26-row
        # hidden_states for a 6-token prefill once one compiled model started
        # serving every prefill length, which vLLM then indexes out of bounds.
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
