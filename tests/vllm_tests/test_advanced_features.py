# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Coverage for the vLLM features called out in the vllm README's Limitations section.

Speculative decoding, beam search, grammar-constrained decoding, custom
logit processors, and `logprobs > 0` are documented as falling back to
vLLM's slow Python sampler (the OV-fused sampler eligibility check rejects
them) -- this file checks that they still produce correct output through
the OV backend even on that fallback path, not that the fallback itself
fires. Continuous batching and prefix caching are documented as unaffected.

Every test builds and deletes its own LLM instance (see new_openvino_llm):
sharing one across tests left it resident for the whole module regardless
of test order, stacking its footprint onto every other test's own peak.
"""

import os

import pytest
from vllm.v1.sample.logits_processor import LogitsProcessor

from conftest import new_openvino_llm

PROMPT = "The capital of France is"


class _BanTokenLogitsProcessor(LogitsProcessor):
    """Module-level (not nested in a test function) so it's picklable across a spawned EngineCore subprocess.

    The banned token id can't be passed through __init__ (vLLM constructs
    this itself from just `(vllm_config, device, is_pin_memory)`), so it's
    read from an env var. That env var must be set on the parent *before*
    the LLM using this processor is constructed -- a spawned child inherits
    the parent's environment at process-creation time, but mutations the
    parent makes afterwards are invisible to an already-running child.
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
    """Own LLM instance (see new_openvino_llm)."""
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
    """Own LLM instance (see new_openvino_llm)."""
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
    """Two own LLM instances, one at a time (see new_openvino_llm).

    vLLM's V1 engine pickles `logits_processors` classes to hand off to a
    spawned EngineCore subprocess -- a class local to a function isn't
    picklable by reference, hence `_BanTokenLogitsProcessor` at module
    scope. The banned token id has to be known before constructing the LLM
    that bans it, so a plain baseline instance is built, used, and deleted
    first, then the logits-processor instance is built separately.
    """
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True)

    baseline_llm = new_openvino_llm()
    try:
        baseline = baseline_llm.generate([{"prompt": PROMPT}], params)[0].outputs[0]
    finally:
        del baseline_llm
    banned_token_id = baseline.token_ids[0]

    # Spawned children inherit the parent's environment at process-creation
    # time; must be set before constructing the LLM below, not after.
    os.environ[_BanTokenLogitsProcessor.BAN_ENV] = str(banned_token_id)
    banned_llm = new_openvino_llm(logits_processors=[_BanTokenLogitsProcessor])
    try:
        banned = banned_llm.generate([{"prompt": PROMPT}], params)[0].outputs[0]
    finally:
        del banned_llm
        os.environ.pop(_BanTokenLogitsProcessor.BAN_ENV, None)

    assert banned.token_ids[0] != banned_token_id


@pytest.mark.precommit
def test_continuous_batching():
    """Own LLM instance: a 3-prompt batch is a new shape (see new_openvino_llm)."""
    from vllm import SamplingParams

    llm = new_openvino_llm()
    try:
        prompts = [
            "Hi",
            "The quick brown fox jumps over the lazy dog near the riverbank at dawn.",
            "1 + 1 =",
        ]
        out = llm.generate(
            [{"prompt": p} for p in prompts],
            SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True),
        )
        assert len(out) == len(prompts)
        assert all(o.outputs[0].text for o in out)
    finally:
        del llm


@pytest.mark.precommit
def test_beam_search():
    """Own LLM instance (see new_openvino_llm).

    max_tokens=2, not 8: beam_search compiles and retains a new OV model per
    decode step (~3.3 GB each, measured), unlike generate()'s single reused
    shape.
    """
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
    """n-gram speculative decoding: needs no draft-model download.

    Own LLM instance (see new_openvino_llm): speculative decoding is
    configured at engine construction, which the default config doesn't set.
    """
    from vllm import SamplingParams

    llm = new_openvino_llm(speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 3,
        "prompt_lookup_max": 4,
        "prompt_lookup_min": 2,
    })
    try:
        # Repeated substring so n-gram lookup has something to match against.
        prompt = "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the"
        out = llm.generate([{"prompt": prompt}], SamplingParams(max_tokens=16, temperature=0.0, ignore_eos=True))
        assert out[0].outputs[0].text.strip()
    finally:
        del llm


@pytest.mark.precommit
def test_prefix_caching():
    """Own LLM instance (see new_openvino_llm): the default config doesn't
    set enable_prefix_caching.

    "One day," / "One night," tokenize to the same length (verified), so all
    three calls share one compiled shape instead of paying for a second one.
    """
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
        # Different suffix, same cached prefix KV blocks.
        text2 = llm.generate([{"prompt": shared_prefix + "One night,"}], params)[0].outputs[0].text
        # Identical prompt to the first call: full cache hit, must be
        # byte-identical (determinism-under-cache-reuse check).
        text3 = llm.generate([{"prompt": shared_prefix + "One day,"}], params)[0].outputs[0].text

        assert text1.strip() and text2.strip()
        assert text1 == text3
    finally:
        del llm
