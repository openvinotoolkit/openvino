# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# mypy: ignore-errors

"""Run lm_head through OV instead of oneDNN (OV_LM_HEAD=1).

Off by default: a second live InferRequest slows the main graph more than the OMP_NUM_THREADS tuning it saves.
"""

import logging
import os

logger = logging.getLogger(__name__)

# Distinct row counts to keep staging buffers for; beyond that, reallocate.
_MAX_CACHED_SHAPES = 32


def _ov_type_for(torch_dtype):
    """Map a torch float dtype to the OV type, or None if unsupported."""
    import openvino as ov
    import torch

    return {
        torch.bfloat16: ov.Type.bf16,
        torch.float16: ov.Type.f16,
        torch.float32: ov.Type.f32,
    }.get(torch_dtype)


def _copy_into(ov_tensor, tensor):
    """Fill an OV tensor from a torch tensor; bf16 goes through uint16 bits, since numpy has no bfloat16."""
    import numpy as np
    import torch

    dst = np.asarray(ov_tensor.data)
    src = tensor.contiguous()
    if src.dtype == torch.bfloat16:
        dst.view(np.uint16)[:] = src.view(torch.uint16).numpy()
    else:
        dst[:] = src.numpy()


def build_ov_lm_head(weight, nthreads=None):
    """Compile ``[M, H] x [H, V]`` on OV as a cpu_linear-shaped callable; None makes the caller use oneDNN."""
    import numpy as np
    import openvino as ov
    import openvino.opset13 as op
    import torch

    et = _ov_type_for(weight.dtype)
    if et is None:
        logger.debug("[OV plugin] lm_head dtype %s not handled by OV path", weight.dtype)
        return None

    if nthreads is None:
        # The same thread pool as the main graph, which never runs concurrently; 0 lets OV decide.
        nthreads = int(os.environ.get("OV_INFERENCE_NUM_THREADS", "0") or 0)

    vocab_size, hidden_size = weight.shape
    wt = ov.Tensor(et, [vocab_size, hidden_size])
    _copy_into(wt, weight)

    # transpose_b: vLLM stores lm_head as [vocab, hidden], and this avoids copying it.
    param = op.parameter([-1, hidden_size], et, name="x")
    model = ov.Model([op.result(op.matmul(param, op.constant(wt), False, True))],
                     [param], "ov_lm_head")

    # The model's own dtype, not the preset's bf16 substitute, which would flip greedy argmax on near-ties.
    cfg = {"INFERENCE_PRECISION_HINT": et.get_type_name()}
    if nthreads:
        cfg["INFERENCE_NUM_THREADS"] = nthreads
    compiled = ov.Core().compile_model(model, "CPU", cfg)

    req = compiled.create_infer_request()
    iport, oport = compiled.inputs[0], compiled.outputs[0]
    staging = {}

    def cpu_linear(inp, weight, bias):
        rows = inp.shape[0]
        buf = staging.get(rows)
        if buf is None:
            if len(staging) >= _MAX_CACHED_SHAPES:
                staging.clear()
            buf = staging[rows] = ov.Tensor(et, [rows, hidden_size])
        _copy_into(buf, inp)
        req.set_tensor(iport, buf)
        req.infer(share_outputs=True)

        out = req.get_tensor(oport).data
        # The next infer() overwrites this buffer, and callers may hold logits across steps.
        out = torch.from_numpy(np.asarray(out)).clone()
        if et == ov.Type.bf16:
            out = out.view(torch.bfloat16)
        return out + bias if bias is not None else out

    logger.info("[OV plugin] lm_head compiled on OV: [%d, %d] %s, %s threads",
                vocab_size, hidden_size, et.get_type_name(), nthreads or "auto")
    return cpu_linear
