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
