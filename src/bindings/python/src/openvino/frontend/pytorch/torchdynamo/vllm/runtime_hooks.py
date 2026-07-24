# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""vLLM-specific runtime hooks for the OV inference path.

Helpers called from torchdynamo.execute to keep the generic infer code
free of vLLM-specific PA-binding knowledge.
"""

from .side_channel import _bind_paged_attention_side_channel


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
