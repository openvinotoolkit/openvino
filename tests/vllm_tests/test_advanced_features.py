# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""vLLM features beyond plain decoding work through OV, also those that fall back to vLLM's Python sampler.

Each test builds and deletes its own LLM (see new_openvino_llm).
"""

import os

import pytest
from vllm.v1.sample.logits_processor import LogitsProcessor

from conftest import new_openvino_llm

PROMPT = "The capital of France is"


class _BanTokenLogitsProcessor(LogitsProcessor):
    """Bans the token in BAN_ENV; module-level so the spawned EngineCore can unpickle it.

    vLLM builds it itself, so the token id comes from the environment, set before the LLM spawns its engine.
    """

    BAN_ENV = "OV_VLLM_TEST_BANNED_TOKEN_ID"

    def __init__(self, vllm_config, device, is_pin_memory):
        pass

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update):
        pass

    def apply(self, logits):
        banned = os.environ.get(self.BAN_ENV)
        if banned:
            logits[:, int(banned)] = float("-inf")
        return logits


@pytest.mark.precommit
def test_logprobs_greater_than_zero():
    from vllm import SamplingParams

    llm = new_openvino_llm()
    try:
        out = llm.generate(
            [{"prompt": PROMPT}],
            SamplingParams(max_tokens=5, temperature=0.0, logprobs=5),
        )
        result = out[0].outputs[0]
        assert result.logprobs is not None and len(result.logprobs) == 5
        assert all(len(step) >= 1 for step in result.logprobs)
    finally:
        del llm


@pytest.mark.precommit
def test_grammar_constrained_decoding():
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = new_openvino_llm()
    try:
        out = llm.generate(
            [{"prompt": PROMPT}],
            SamplingParams(
                max_tokens=6, temperature=0.0,
                structured_outputs=StructuredOutputsParams(regex=r"[0-9]+"),
            ),
        )
        text = out[0].outputs[0].text.strip()
        assert text and all(c.isdigit() for c in text)
    finally:
        del llm


@pytest.mark.precommit
def test_custom_logits_processor():
    """A baseline LLM picks the token to ban; a second LLM, built after, must not emit it."""
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True)

    baseline_llm = new_openvino_llm()
    try:
        baseline = baseline_llm.generate([{"prompt": PROMPT}], params)[0].outputs[0]
    finally:
        del baseline_llm
    banned_token_id = baseline.token_ids[0]

    os.environ[_BanTokenLogitsProcessor.BAN_ENV] = str(banned_token_id)
    banned_llm = new_openvino_llm(logits_processors=[_BanTokenLogitsProcessor])
    try:
        banned = banned_llm.generate([{"prompt": PROMPT}], params)[0].outputs[0]
    finally:
        del banned_llm
        os.environ.pop(_BanTokenLogitsProcessor.BAN_ENV, None)

    assert banned.token_ids[0] != banned_token_id


@pytest.mark.precommit
def test_varying_shapes_and_continuous_batching():
    """Prefill lengths that change across generate() calls, then a mixed batch, on one model."""
    from vllm import SamplingParams

    long_prompt = "The quick brown fox jumps over the lazy dog near the riverbank at dawn."
    params = SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True)
    llm = new_openvino_llm()
    try:
        for prompts in (["Hi"], [long_prompt], ["Hi", long_prompt, "1 + 1 ="]):
            out = llm.generate([{"prompt": p} for p in prompts], params)
            assert len(out) == len(prompts)
            for prompt, o in zip(prompts, out):
                assert len(o.outputs[0].token_ids) == 8 and o.outputs[0].text, f"{prompt!r}: {o.outputs[0]}"
    finally:
        del llm


@pytest.mark.precommit
def test_beam_search():
    """max_tokens=2: beam_search compiles and keeps a new OV model per decode step."""
    from vllm.sampling_params import BeamSearchParams

    llm = new_openvino_llm()
    try:
        out = llm.beam_search(
            [{"prompt": PROMPT}],
            BeamSearchParams(beam_width=3, max_tokens=2),
        )
        sequences = out[0].sequences
        assert len(sequences) == 3
        assert all(s.text for s in sequences)
    finally:
        del llm


@pytest.mark.precommit
def test_speculative_decoding_ngram():
    """n-gram speculative decoding, which needs no draft model."""
    from vllm import SamplingParams

    llm = new_openvino_llm(speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 3,
        "prompt_lookup_max": 4,
        "prompt_lookup_min": 2,
    })
    try:
        # The repeat gives the n-gram lookup something to match.
        prompt = "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the"
        out = llm.generate([{"prompt": prompt}], SamplingParams(max_tokens=16, temperature=0.0, ignore_eos=True))
        assert out[0].outputs[0].text.strip()
    finally:
        del llm


@pytest.mark.precommit
def test_prefix_caching():
    """The suffixes tokenize to the same length, so all three calls share one compiled shape."""
    from vllm import SamplingParams

    llm = new_openvino_llm(enable_prefix_caching=True)
    try:
        shared_prefix = (
            "In a small village surrounded by mountains, there lived an old "
            "clockmaker who believed every gear told a story about the people "
            "who once needed it. "
        )
        params = SamplingParams(max_tokens=4, temperature=0.0, ignore_eos=True)

        text1 = llm.generate([{"prompt": shared_prefix + "One day,"}], params)[0].outputs[0].text
        text2 = llm.generate([{"prompt": shared_prefix + "One night,"}], params)[0].outputs[0].text
        # A full cache hit must reproduce the first output.
        text3 = llm.generate([{"prompt": shared_prefix + "One day,"}], params)[0].outputs[0].text

        assert text1.strip() and text2.strip()
        assert text1 == text3
    finally:
        del llm
