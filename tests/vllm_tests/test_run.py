# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OV vs vLLM eager: byte-identical greedy output on TinyLlama, and the worker state the plugin sets.

float32 with the OV precision overrides: in bf16 eager, Inductor and OV disagree from the first token.
"""

import pytest

from conftest import MODEL_ID, select_cpu_platform

PROMPT = "The capital of France is "
MAX_NEW_TOKENS = 32
# A fixed KV pool skips vLLM's 92%-free-memory check, so this fits 16 GB machines.
KV_CACHE_BYTES = 512 << 20


def _worker_kv_update_flags(worker):
    """Run in the engine worker: (stock CPUAttentionBackend flag, sorted per-layer flags of the loaded model)."""
    from vllm.v1.attention.backends.cpu_attn import CPUAttentionBackend
    from vllm.v1.attention.backend import AttentionBackend

    model = worker.model_runner.get_model()
    layer_flags = {
        m.attn_backend.forward_includes_kv_cache_update
        for m in model.modules()
        # Encoder layers hold an AttentionBackendEnum, not a backend class.
        if (
            isinstance(getattr(m, "attn_backend", None), type)
            and issubclass(m.attn_backend, AttentionBackend)
        )
    }
    return CPUAttentionBackend.forward_includes_kv_cache_update, sorted(layer_flags)


def _worker_omp_wait_policy(worker):
    """OMP_WAIT_POLICY as seen by the engine worker."""
    import os

    return os.environ.get("OMP_WAIT_POLICY")


def _run(llm):
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0, ignore_eos=True)
    llm.generate([{"prompt": PROMPT}], params)  # compiles the model
    return llm.generate([{"prompt": PROMPT}], params)[0].outputs[0].text


@pytest.mark.precommit
def test_openvino_matches_eager_greedy(monkeypatch):
    select_cpu_platform()
    monkeypatch.setenv("OV_INFERENCE_PRECISION_HINT", "f32")
    monkeypatch.setenv("OV_KV_CACHE_PRECISION", "f32")
    # The plugin only fills OMP_WAIT_POLICY in when it is unset.
    monkeypatch.delenv("OMP_WAIT_POLICY", raising=False)
    from vllm import LLM

    eager_llm = LLM(
        model=MODEL_ID,
        dtype="float32",
        enforce_eager=True,
        max_model_len=2048,
        distributed_executor_backend="uni",
        kv_cache_memory_bytes=KV_CACHE_BYTES,
    )
    try:
        eager_text = _run(eager_llm)
        eager_wait_policy = eager_llm.collective_rpc(_worker_omp_wait_policy)
    finally:
        del eager_llm

    ov_llm = LLM(
        model=MODEL_ID,
        dtype="float32",
        enforce_eager=False,
        max_model_len=2048,
        distributed_executor_backend="uni",
        block_size=32,
        kv_cache_memory_bytes=KV_CACHE_BYTES,
        compilation_config={
            "mode": "STOCK_TORCH_COMPILE",
            "backend": "openvino",
            "custom_ops": ["none"],
        },
    )
    try:
        ov_text = _run(ov_llm)
        kv_update_flags = ov_llm.collective_rpc(_worker_kv_update_flags)
        ov_wait_policy = ov_llm.collective_rpc(_worker_omp_wait_policy)
    finally:
        del ov_llm

    assert eager_text == ov_text, (
        f"OV and eager outputs diverge:\n  eager:    {eager_text!r}\n  openvino: {ov_text!r}")

    # OV owns the KV-cache write on its own layers only; flipping the class flag would disarm the write for every
    # other model in the worker (a draft model, a second in-process engine).
    stock_flag, layer_flags = kv_update_flags[0]
    assert stock_flag is False, (
        "OV load mutated CPUAttentionBackend.forward_includes_kv_cache_update "
        "process-wide in the engine worker")
    assert layer_flags == [True], (
        f"OV model's attention layers did not take over the KV-cache write: {layer_flags}")

    # OV workers must not let libgomp spin on the cores TBB is pinned to.
    assert ov_wait_policy == ["PASSIVE"], f"OV worker OMP_WAIT_POLICY: {ov_wait_policy}"
    assert eager_wait_policy == [None], f"eager worker OMP_WAIT_POLICY: {eager_wait_policy}"
