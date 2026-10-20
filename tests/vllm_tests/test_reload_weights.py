# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Verify the OV model picks up weights changed in place after load.

AOT freezing bakes every weight into the OV model as a constant, and vLLM's
reload_weights copies new values into the existing Parameters (same address,
same shape). Unless the plugin recompiles, the OV model keeps serving the
pre-reload weights while eager vLLM would already use the new ones.
"""

import pytest
import torch

from conftest import new_openvino_llm

PROMPT = "The capital of France is"


def _zero_down_proj(worker):
    """Shipped to the engine worker by `LLM.collective_rpc`: reload every MLP down projection as zeros."""
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
    """Shipped to the engine worker: the number of OV models alive there."""
    import gc

    import openvino as ov
    gc.collect()
    return sum(isinstance(o, ov.CompiledModel) for o in gc.get_objects())


@pytest.mark.precommit
def test_reload_weights_reaches_ov_model():
    """Zeroing the MLP outputs must change the generated text; a stale OV model would repeat it."""
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
    # The rebuild replays vLLM's warmup like the load does; without it the first request
    # compiles an extra static-shape OV model (with its own copy of the weights).
    assert models_after == models_before, f"OV models alive: {models_before} before reload, {models_after} after"


@pytest.mark.precommit
def test_clear_caches_drops_unpacked_moe_weights():
    """The unpacked MoE expert copies are keyed by the packed weight's address, which in-place reloads keep."""
    from openvino.frontend.pytorch.torchdynamo import execute
    from openvino.frontend.pytorch.torchdynamo.vllm import fused_moe

    plain = torch.zeros(2, 4, 8)
    fused_moe._unpacked_cache[(plain.data_ptr(), tuple(plain.shape), "amx")] = plain
    execute.clear_caches()
    assert len(fused_moe._unpacked_cache) == 0
