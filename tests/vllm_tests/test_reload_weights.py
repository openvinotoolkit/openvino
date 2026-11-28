# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""In-place weight reloads must reach the OV model, which holds the weights as constants."""

import pytest
import torch

from conftest import new_openvino_llm

PROMPT = "The capital of France is"


def _zero_down_proj(worker):
    """Run in the engine worker: reload every MLP down projection as zeros."""
    runner = worker.model_runner
    model = runner.get_model()
    cfg = runner.model_config.hf_config
    weights = [(f"model.layers.{i}.mlp.down_proj.weight",
                torch.zeros(cfg.hidden_size, cfg.intermediate_size, dtype=torch.bfloat16))
               for i in range(len(model.model.layers))]
    forward = model.forward
    runner.reload_weights(weights_iterator=iter(weights))
    return model.forward is not forward


def _live_compiled_models(worker):
    """Run in the engine worker: the number of live OV compiled models."""
    import gc

    import openvino as ov
    gc.collect()
    return sum(isinstance(o, ov.CompiledModel) for o in gc.get_objects())


@pytest.mark.precommit
def test_reload_weights_reaches_ov_model():
    """Zeroed MLP outputs must change the text; a stale OV model would repeat it."""
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=16)
    llm = new_openvino_llm(max_model_len=512)
    try:
        before = llm.generate([PROMPT], params)[0].outputs[0].text
        models_before = llm.collective_rpc(_live_compiled_models)
        recompiled = llm.collective_rpc(_zero_down_proj)
        after = llm.generate([PROMPT], params)[0].outputs[0].text
        models_after = llm.collective_rpc(_live_compiled_models)
    finally:
        del llm
    assert all(recompiled), "reload_weights did not recompile the OV forward"
    assert after != before, f"OV output did not change after reload_weights: {after!r}"
    # Without replaying the warmup, the next request compiles an extra static-shape model.
    assert models_after == models_before, f"OV models alive: {models_before} before reload, {models_after} after"


@pytest.mark.precommit
def test_clear_caches_drops_unpacked_moe_weights():
    """The unpacked MoE weights are keyed by the packed address, which an in-place reload keeps."""
    from openvino.frontend.pytorch.torchdynamo import execute
    from openvino.frontend.pytorch.torchdynamo.vllm import fused_moe

    plain = torch.zeros(2, 4, 8)
    key = (plain.data_ptr(), tuple(plain.shape), "amx")
    fused_moe._unpacked_cache[key] = plain
    halves = [plain[:, :2].contiguous(), plain[:, 2:].contiguous()]
    for i, h in enumerate(halves):
        fused_moe._half_cache[key + (i,)] = h
    execute.clear_caches()
    assert len(fused_moe._unpacked_cache) == 0
    assert len(fused_moe._half_cache) == 0
