# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OV backend vs vLLM eager on tiny random-weight MoE and Gated DeltaNet models.

The generation tests load dense TinyLlama, so neither the MoE rewrite
(fused_moe.rewrite_cpu_fused_moe, lowered to GatherMatmul) nor the GDN
rewrite (gated_delta_net.rewrite_gdn_to_ov, lowered to PagedGatedDeltaNet with
its convolution/recurrent state written back each step) runs there. Each
case writes a tiny random checkpoint with transformers, runs a prefill plus
16 greedy decode steps through vLLM eager and through the OV backend, and
compares the tokens and their logprobs step by step. Decode steps start from
the state the previous step wrote back, so a state-layout or writeback bug
shows up as a divergence at the first decode step.

Random tiny models have flat logits, so some steps are near ties that
rounding alone can flip. A token mismatch is only accepted where eager's own
top-1/top-2 margin is within the tolerance; the comparison stops there, after
at least MIN_STEPS matching steps.

Qwen3.5 (GDN) runs in bfloat16: vLLM's CPU causal_conv1d kernel has no
float32 implementation. The MoE models run in float32 (see test_run.py for
why that needs the OV precision overrides).

No model download needed.
"""

import pytest
import torch

from conftest import select_cpu_platform

# Generated tokens compared per case, after the prefill.
DECODE_STEPS = 16
MIN_STEPS = 4
KV_CACHE_BYTES = 256 << 20

_GEMMA4_ROPE = {
    "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
    "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
}
_QWEN3_5_ROPE = {
    "mrope_interleaved": True, "mrope_section": [11, 11, 10], "rope_type": "default",
    "rope_theta": 10000000, "partial_rotary_factor": 0.25,
}

# name: (model_type, config, dtype, block_size, prompt seed/length, logprob tolerance, OV layer the rewrite lowers to)
CASES = {
    "qwen3_moe_silu": (
        "qwen3_moe",
        dict(architectures=["Qwen3MoeForCausalLM"], hidden_size=256, intermediate_size=512,
             moe_intermediate_size=128, num_experts=8, num_experts_per_tok=2, decoder_sparse_step=1,
             norm_topk_prob=True, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
             hidden_act="silu", rms_norm_eps=1e-6, vocab_size=1024, tie_word_embeddings=False,
             max_position_embeddings=4096, rope_theta=1000000.0),
        "float32", 32, (15, 29), 1e-3, "GatherMatmul"),
    "gemma4_moe_gelu_tanh": (
        "gemma4_text",
        dict(architectures=["Gemma4ForCausalLM"], hidden_size=256, intermediate_size=512, moe_intermediate_size=128,
             num_experts=8, top_k_experts=2, enable_moe_block=True, num_hidden_layers=2,
             layer_types=["sliding_attention", "full_attention"], num_attention_heads=2, num_key_value_heads=1,
             num_global_key_value_heads=1, head_dim=256, global_head_dim=512, attention_k_eq_v=True,
             hidden_activation="gelu_pytorch_tanh", sliding_window=64, final_logit_softcapping=30.0,
             rms_norm_eps=1e-6, vocab_size=1024, vocab_size_per_layer_input=1024, hidden_size_per_layer_input=0,
             num_kv_shared_layers=0, tie_word_embeddings=True, max_position_embeddings=4096,
             use_double_wide_mlp=False, pad_token_id=0, bos_token_id=2, eos_token_id=1,
             rope_parameters=_GEMMA4_ROPE),
        "float32", 64, (5, 34), 1e-3, "GatherMatmul"),
    "qwen3_5_gdn": (
        "qwen3_5_text",
        # initializer_range 0.1: at the default 0.02 the GDN branch barely moves the residual stream, so a
        # dropped state writeback stays within bf16 noise.
        dict(architectures=["Qwen3_5ForCausalLM"], hidden_size=256, intermediate_size=512, num_hidden_layers=4,
             layer_types=["linear_attention"] * 3 + ["full_attention"], full_attention_interval=4,
             num_attention_heads=2, num_key_value_heads=1, head_dim=256, attn_output_gate=True,
             linear_conv_kernel_dim=4, linear_key_head_dim=128, linear_value_head_dim=128,
             linear_num_key_heads=2, linear_num_value_heads=2, hidden_act="silu", rms_norm_eps=1e-6,
             vocab_size=1024, tie_word_embeddings=False, initializer_range=0.1, max_position_embeddings=4096,
             mamba_ssm_dtype="float32", mtp_num_hidden_layers=0, rope_parameters=_QWEN3_5_ROPE),
        "bfloat16", 32, (12, 33), 0.15, "PagedGatedDeltaNet"),
}


def _prompt(seed, length):
    return [(seed * 131 + i * (17 + seed)) % 1000 + 10 for i in range(length)]


def _write_checkpoint(path, model_type, config):
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(AutoConfig.for_model(model_type, **config), torch_dtype=torch.float32)
    model.save_pretrained(path)


def _worker_layer_types(worker):
    """Count the layer types of every compiled OV model in the engine worker's cache."""
    from collections import Counter

    from openvino.frontend.pytorch.torchdynamo import execute

    counts = Counter()
    for compiled in list(execute.compiled_cache.values()):
        for op in compiled.get_runtime_model().get_ordered_ops():
            counts[op.get_rt_info()["layerType"].astype(str)] += 1
    return dict(counts)


def _generate(path, dtype, prompt, openvino, block_size):
    from vllm import LLM, SamplingParams

    kwargs = dict(model=path, dtype=dtype, max_model_len=512, distributed_executor_backend="uni",
                  skip_tokenizer_init=True, kv_cache_memory_bytes=KV_CACHE_BYTES, max_num_seqs=1,
                  enable_prefix_caching=False)
    if openvino:
        kwargs.update(enforce_eager=False, block_size=block_size, compilation_config={
            "mode": "STOCK_TORCH_COMPILE", "backend": "openvino", "custom_ops": ["none"]})
    else:
        kwargs.update(enforce_eager=True)
    llm = LLM(**kwargs)
    try:
        params = SamplingParams(max_tokens=DECODE_STEPS, temperature=0.0, ignore_eos=True, logprobs=3,
                                detokenize=False)
        llm.generate([{"prompt_token_ids": prompt}], params, use_tqdm=False)  # warmup: compiles the model
        out = llm.generate([{"prompt_token_ids": prompt}], params, use_tqdm=False)[0].outputs[0]
        logprobs = [{token: lp.logprob for token, lp in step.items()} for step in out.logprobs]
        layers = llm.collective_rpc(_worker_layer_types)[0] if openvino else {}
    finally:
        del llm
    return list(out.token_ids), logprobs, layers


@pytest.mark.precommit
@pytest.mark.parametrize("case", list(CASES))
def test_openvino_matches_eager_tiny(case, tmp_path, monkeypatch):
    model_type, config, dtype, block_size, prompt_spec, tol, ov_layer = CASES[case]
    select_cpu_platform()
    if dtype == "float32":
        monkeypatch.setenv("OV_INFERENCE_PRECISION_HINT", "f32")
        monkeypatch.setenv("OV_KV_CACHE_PRECISION", "f32")
    _write_checkpoint(tmp_path, model_type, config)
    prompt = _prompt(*prompt_spec)

    eager_tokens, eager_lps, _ = _generate(str(tmp_path), dtype, prompt, False, block_size)
    ov_tokens, ov_lps, layers = _generate(str(tmp_path), dtype, prompt, True, block_size)

    assert layers.get(ov_layer), f"{ov_layer} missing from the OV exec graphs: {layers}"
    assert len(eager_tokens) == len(ov_tokens) == DECODE_STEPS, (
        f"expected {DECODE_STEPS} tokens, got eager {len(eager_tokens)} and OV {len(ov_tokens)}")
    _assert_same_decode(eager_tokens, eager_lps, ov_tokens, ov_lps, tol)


def _assert_same_decode(eager_tokens, eager_lps, ov_tokens, ov_lps, tol):
    """Tokens and their logprobs agree step by step; a mismatch is only accepted at a near tie, then comparison stops.

    A near tie means each side ranks the other's token within `tol` of its own pick, so both tokens must appear in
    both sides' logprobs (vLLM always returns the sampled token's logprob alongside the top ones).
    """
    lengths = (len(eager_tokens), len(eager_lps), len(ov_tokens), len(ov_lps))
    assert len(set(lengths)) == 1 and lengths[0] > MIN_STEPS, f"token/logprob lengths {lengths}"
    for step, (eager_token, ov_token) in enumerate(zip(eager_tokens, ov_tokens)):
        if eager_token != ov_token:
            context = (f"step {step}: OV picked {ov_token}, eager {eager_token}\n"
                       f"  eager: {eager_tokens}\n  ov:    {ov_tokens}")
            for side, lps, pick, other in (("eager", eager_lps, eager_token, ov_token),
                                           ("OV", ov_lps, ov_token, eager_token)):
                assert other in lps[step], f"{context}\n  {side} doesn't rank token {other} among its candidates"
                gap = lps[step][pick] - lps[step][other]
                assert gap <= tol, f"{context}\n  {side} prefers its token by {gap:.4f} (tolerance {tol})"
            assert step >= MIN_STEPS, f"diverged at a near tie after only {step} steps"
            return
        assert eager_token in ov_lps[step], f"step {step}: token {eager_token} missing from OV logprobs"
        delta = abs(eager_lps[step][eager_token] - ov_lps[step][eager_token])
        assert delta <= tol, f"step {step}: logprob differs by {delta:.4f} (tolerance {tol})"


@pytest.mark.precommit
def test_near_tie_acceptance():
    """The comparison accepts a divergence only between near-tied tokens, never a far-off pick."""
    match = ([5] * MIN_STEPS, [{5: -0.1, 6: -3.0}] * MIN_STEPS)
    eager_tokens, eager_lps = match[0] + [1], match[1] + [{1: -1.0, 2: -1.0005, 900: -20.0}]
    tied = ([2], [{2: -1.0, 1: -1.0004, 7: -5.0}])
    far = ([900], [{900: -0.9, 1: -1.0, 2: -1.1}])
    _assert_same_decode(eager_tokens, eager_lps, match[0] + tied[0], match[1] + tied[1], tol=1e-3)
    with pytest.raises(AssertionError, match="eager prefers its token"):
        _assert_same_decode(eager_tokens, eager_lps, match[0] + far[0], match[1] + far[1], tol=1e-3)
    with pytest.raises(AssertionError, match="OV prefers its token"):
        unsure_ov = ([2], [{2: -0.5, 1: -1.0, 7: -5.0}])
        _assert_same_decode(eager_tokens, eager_lps, match[0] + unsure_ov[0], match[1] + unsure_ov[1], tol=1e-3)
    with pytest.raises(AssertionError, match="after only"):
        early = ([1] + match[0], eager_lps[-1:] + match[1])
        _assert_same_decode(*early, tied[0] + match[0], tied[1] + match[1], tol=1e-3)
    with pytest.raises(AssertionError, match="lengths"):
        _assert_same_decode(eager_tokens, eager_lps, match[0], match[1], tol=1e-3)  # truncated OV output
    with pytest.raises(AssertionError, match="lengths"):
        _assert_same_decode([], [], [], [], tol=1e-3)


@pytest.mark.precommit
def test_alibi_model_is_refused(tmp_path, monkeypatch):
    """A model with ALiBi attention fails at load with a clear error, instead of running OV attention without the
    slopes. In-process engine, so the worker's exception reaches the test unwrapped."""
    from vllm import LLM

    select_cpu_platform()
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    _write_checkpoint(tmp_path, "bloom", dict(architectures=["BloomForCausalLM"], hidden_size=256, n_layer=2,
                                              n_head=4, vocab_size=1024, layer_norm_epsilon=1e-5))
    with pytest.raises(NotImplementedError, match="ALiBi"):
        LLM(model=str(tmp_path), dtype="float32", max_model_len=512, distributed_executor_backend="uni",
            skip_tokenizer_init=True, kv_cache_memory_bytes=KV_CACHE_BYTES, max_num_seqs=1, block_size=32,
            compilation_config={"mode": "STOCK_TORCH_COMPILE", "backend": "openvino", "custom_ops": ["none"]})


def _routed_reference(tokens, w1, w3, w2, topk_weights, topk_ids, act):
    """Per-token, per-selected-expert MoE: weight * down(act(gate(x)) * up(x))."""
    from torch.nn import functional

    activations = {
        "silu": functional.silu,
        "gelu": functional.gelu,
        "gelu_tanh": lambda t: functional.gelu(t, approximate="tanh"),
    }
    out = torch.zeros_like(tokens)
    for t in range(tokens.shape[0]):
        for weight, expert in zip(topk_weights[t], topk_ids[t]):
            hidden = activations[act](w1[expert] @ tokens[t]) * (w3[expert] @ tokens[t])
            out[t] += weight * (w2[expert] @ hidden)
    return out


@pytest.mark.precommit
@pytest.mark.parametrize("act", ["silu", "gelu", "gelu_tanh"])
def test_moe_tiled_matches_routed_reference(act):
    """moe_tiled's math, which the OV graph keeps whether or not GatherMatmul fusion fires, against plain routing."""
    from openvino.frontend.pytorch.torchdynamo.vllm.fused_moe import moe_tiled

    generator = torch.Generator().manual_seed(0)
    num_tokens, hidden, inter, experts, top_k = 5, 16, 24, 6, 2
    tokens = torch.randn(num_tokens, hidden, generator=generator)
    w1 = torch.randn(experts, inter, hidden, generator=generator) * 0.3
    w3 = torch.randn(experts, inter, hidden, generator=generator) * 0.3
    w2 = torch.randn(experts, hidden, inter, generator=generator) * 0.3
    router = torch.randn(num_tokens, experts, generator=generator).softmax(-1)
    topk_weights, topk_ids = router.topk(top_k, dim=-1)

    expected = _routed_reference(tokens, w1, w3, w2, topk_weights, topk_ids, act)
    got = moe_tiled(tokens, w1, w3, w2, topk_weights, topk_ids, act)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)
