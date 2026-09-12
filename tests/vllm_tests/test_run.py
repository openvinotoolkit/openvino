# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: vLLM + OpenVINO backend vs vLLM eager.

Greedy decode only (HuggingFace's `do_sample=False`; vLLM expresses it as
`temperature=0`). No sampling code path is exercised: the eager and OV runs
both go through `Sampler.greedy_sample`, which is a plain argmax over
logits, so any output divergence is attributable to the model.forward
implementation alone.

Uses dtype=float32, not the project-wide bfloat16 default: at bf16, eager,
Inductor, and the OV backend all disagree with each other from the very
first generated token for this model/prompt (verified manually) -- a
generic compiled-vs-eager bf16 rounding difference, not an OV bug (see
"match" note in the vllm README). float32 is the only dtype where all three
backends agree, so it's the only regime where byte-for-byte equality is a
meaningful correctness signal here. Both paths share a process so the
comparison is meaningful (same tokenizer state, same prompt encoding); the
eager and OV `LLM` instances are never alive at the same time.
"""

import pytest

from conftest import MODEL_ID, select_cpu_platform

PROMPT = "The capital of France is "
MAX_NEW_TOKENS = 32
SKIP_WARMUP_TOKENS = 5


def _generate(llm, prompt, params):
    out = llm.generate([{"prompt": prompt}], params)
    return out[0].outputs[0]


def _run(llm, prompt, max_new_tokens, skip_warmup_tokens):
    from vllm import SamplingParams

    # temperature=0 (do_sample=False) -> greedy argmax; no sampling code
    # path is exercised.
    warm_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0, ignore_eos=True)
    _generate(llm, prompt, warm_params)  # warmup: compiles the model

    full_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0, ignore_eos=True)
    out = _generate(llm, prompt, full_params)
    return out.text


@pytest.mark.precommit
def test_openvino_matches_eager_greedy():
    """The OV backend and vLLM eager must produce byte-identical greedy output.

    Builds its own float32 LLM instances rather than using the shared
    `openvino_llm` fixture (bfloat16): see module docstring for why bf16
    isn't a stable regime for this equality check.
    """
    select_cpu_platform()
    from vllm import LLM

    eager_llm = LLM(
        model=MODEL_ID,
        dtype="float32",
        enforce_eager=True,
        max_model_len=2048,
        distributed_executor_backend="uni",
    )
    try:
        eager_text = _run(eager_llm, PROMPT, MAX_NEW_TOKENS, SKIP_WARMUP_TOKENS)
    finally:
        del eager_llm

    ov_llm = LLM(
        model=MODEL_ID,
        dtype="float32",
        enforce_eager=False,
        max_model_len=2048,
        distributed_executor_backend="uni",
        block_size=32,
        compilation_config={
            "mode": "STOCK_TORCH_COMPILE",
            "backend": "openvino",
            "custom_ops": ["none"],
        },
    )
    try:
        ov_text = _run(ov_llm, PROMPT, MAX_NEW_TOKENS, SKIP_WARMUP_TOKENS)
    finally:
        del ov_llm

    assert eager_text == ov_text, (
        f"OV and eager outputs diverge:\n  eager:    {eager_text!r}\n  openvino: {ov_text!r}")
