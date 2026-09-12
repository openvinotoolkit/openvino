# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run lm_head through OV instead of torch, so it stops depending on OMP_NUM_THREADS.

lm_head is the one heavy torch op left in the decode step: vLLM v1 calls
``compute_logits()`` separately from ``forward()``, outside the OV-compiled
region, so its ``[M, hidden] x [hidden, vocab]`` GEMM runs on torch's pool
while the model runs on OV's. That split is deliberate -- vLLM gathers
``logits_indices`` in between so lm_head sees only the sampled rows -- so the
fix is not to fold lm_head into the traced graph but to swap the leaf callable
``layer.cpu_linear`` (vllm/model_executor/layers/utils.py).

The motivation is *whose* thread pool it lands on, not kernel speed: at equal
threads oneDNN's AMX-prepacked path beats OV's FullyConnected (127 vs 148 us
TinyLlama-1.1B, 794 vs 866 us Qwen2.5-1.5B, 60 threads). But this path keeps
torch narrow so its OMP workers do not spin-wait against OV's TBB pool, and at
OMP_NUM_THREADS=2 that same oneDNN GEMM costs 2066 / 11266 us -- 16x and 13x
worse. On OV it gets the full pool whatever OMP_NUM_THREADS says.

That removes the knob rather than retuning it. On oneDNN the best
OMP_NUM_THREADS varies with model *and* core count (4/4/8/8 at 8/16/32/60
cores for TinyLlama, 8/16/16 for Qwen2.5-1.5B), cannot be discovered at
runtime -- CPUModelRunner replaces ``torch.set_num_threads`` with a no-op after
thread binding -- and guessing wrong costs up to 28%. On OV, OMP_NUM_THREADS=1
won at every core count and model measured, which is what vLLM's own
``set_torch_threads_for_runtime()`` wants to set anyway.

Opt-in only: this path is OFF by default (OV_LM_HEAD=0), because the win above
is outweighed by a cost the isolated kernel timings do not show. Installing it
adds a second compiled OV model whose InferRequest is invoked between
main-graph infers, and that interleaving roughly doubles the *main graph's*
per-step time -- reproduced on Llama-3.2-1B and Mistral-7B. The penalty is not
in the lm_head GEMM (still competitive, as above) and survives capping the
second model's thread count, disabling its CPU pinning, and forcing both
models onto one ov.Core, so it is not tunable from this layer; it looks like
per-InferRequest state in the CPU plugin. Until that is fixed, tuning
OMP_NUM_THREADS for oneDNN costs less than the interleaving does.

Set OV_LM_HEAD=1 to enable this path. Doing so makes OMP_NUM_THREADS
irrelevant, per the paragraphs above -- useful where that knob cannot be
tuned per model and per machine.
"""

import logging
import os

logger = logging.getLogger(__name__)

# Distinct row-counts to keep staging buffers for. Decode is one row per running
# sequence, so this covers the batch sizes actually seen; past it we realloc
# rather than grow without bound, as varying prefill chunk sizes would.
_MAX_CACHED_SHAPES = 32


def _ov_type_for(torch_dtype):
    """Map a torch float dtype to the OV type, or None if we should not try."""
    import openvino as ov
    import torch

    return {
        torch.bfloat16: ov.Type.bf16,
        torch.float16: ov.Type.f16,
        torch.float32: ov.Type.f32,
    }.get(torch_dtype)


def _copy_into(ov_tensor, t):
    """Fill an OV tensor from a torch tensor, preserving bf16 bits.

    numpy has no bfloat16 and OV surfaces a bf16 buffer as f16 elements of the
    same width, so bf16 has to go through a uint16 re-tag on both sides. Same
    trick as execute._torch_to_numpy, but copying into a pre-declared bf16
    tensor: wrapping an ndarray instead would declare the port f16 and the
    infer request would reject it.
    """
    import numpy as np
    import torch

    dst = np.asarray(ov_tensor.data)
    src = t.contiguous()
    if src.dtype == torch.bfloat16:
        dst.view(np.uint16)[:] = src.view(torch.uint16).numpy()
    else:
        dst[:] = src.numpy()


def build_ov_lm_head(weight, nthreads=None):
    """Compile ``[M, H] x [H, V]`` on OV and return a cpu_linear-shaped callable.

    Returns None if the dtype is not one we handle or anything fails, so the
    caller can fall back to the oneDNN dispatch.
    """
    import numpy as np
    import openvino as ov
    import openvino.opset13 as op
    import torch

    et = _ov_type_for(weight.dtype)
    if et is None:
        logger.debug("[OV plugin] lm_head dtype %s not handled by OV path", weight.dtype)
        return None

    if nthreads is None:
        # Same pool the main graph gets; the two never run concurrently, so
        # there is nothing to share it with. 0 means "let OV decide".
        nthreads = int(os.environ.get("OV_INFERENCE_NUM_THREADS", "0") or 0)

    V, H = weight.shape
    wt = ov.Tensor(et, [V, H])
    _copy_into(wt, weight)

    # transpose_b: vLLM stores lm_head as [vocab, hidden] like F.linear does,
    # so consume b transposed rather than materialize a second copy of a
    # 131-467 MB weight.
    p = op.parameter([-1, H], et, name="x")
    model = ov.Model([op.result(op.matmul(p, op.constant(wt), False, True))],
                     [p], "ov_lm_head")

    # Compute at the model's own dtype, NOT at preset.precision_config()'s
    # choice. That helper substitutes f16 -> bf16 for the main graph because
    # vLLM's unfused RMSNorm squares activations and overflows f16's 65504
    # ceiling. A plain matmul has no such reduction, so here the substitution
    # buys nothing and costs two mantissa bits (bf16 has 8, f16 has 10) --
    # enough to flip greedy argmax on near-ties and make f16 models diverge
    # from eager, which the oneDNN path it replaced did not.
    cfg = {"INFERENCE_PRECISION_HINT": et.get_type_name()}
    if nthreads:
        cfg["INFERENCE_NUM_THREADS"] = nthreads
    compiled = ov.Core().compile_model(model, "CPU", cfg)

    req = compiled.create_infer_request()
    iport, oport = compiled.inputs[0], compiled.outputs[0]
    staging = {}

    def cpu_linear(x, weight, bias):
        rows = x.shape[0]
        buf = staging.get(rows)
        if buf is None:
            if len(staging) >= _MAX_CACHED_SHAPES:
                staging.clear()
            buf = staging[rows] = ov.Tensor(et, [rows, H])
        _copy_into(buf, x)
        req.set_tensor(iport, buf)
        req.infer()

        out = req.get_tensor(oport).data
        # .clone(): `data` views the request's output buffer, which the next
        # infer() overwrites in place. Callers may hold the logits across steps
        # (spec decode does), so this must not alias.
        out = torch.from_numpy(np.asarray(out)).clone()
        if et == ov.Type.bf16:
            out = out.view(torch.bfloat16)
        return out + bias if bias is not None else out

    logger.info("[OV plugin] lm_head compiled on OV: [%d, %d] %s, %s threads",
                V, H, et.get_type_name(), nthreads or "auto")
    return cpu_linear
