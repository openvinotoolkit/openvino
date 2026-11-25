# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""vLLM features beyond plain decoding work through OV, also those that fall back to vLLM's Python sampler.

The tests share one LLM per module: building one costs about a minute, and its memory stays flat as shapes are added.
"""

import gc

import pytest
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

from conftest import new_openvino_llm

PROMPT = "The capital of France is"


class _BanTokenLogitsProcessor(AdapterLogitsProcessor):
    """Bans the token in a request's extra_args["ban_token_id"].

    Module-level, so the spawned EngineCore can unpickle it.
    """

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params):
        banned = (params.extra_args or {}).get("ban_token_id")
        if banned is None:
            return None

        def _ban(output_ids, logits):
            logits[banned] = float("-inf")
            return logits
        return _ban


@pytest.fixture(scope="module")
def llm():
    """Every test here runs greedy or on the Python sampler, so the logits processor changes nothing for the others."""
    llm = new_openvino_llm(logits_processors=[_BanTokenLogitsProcessor])
    yield llm
    del llm
    gc.collect()  # before the next module builds its own


@pytest.mark.precommit
def test_logprobs_greater_than_zero(llm):
    from vllm import SamplingParams

    out = llm.generate(
        [{"prompt": PROMPT}],
        SamplingParams(max_tokens=5, temperature=0.0, logprobs=5),
    )
    result = out[0].outputs[0]
    assert result.logprobs is not None and len(result.logprobs) == 5
    assert all(len(step) >= 1 for step in result.logprobs)


@pytest.mark.precommit
def test_grammar_constrained_decoding(llm):
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    out = llm.generate(
        [{"prompt": PROMPT}],
        SamplingParams(
            max_tokens=6, temperature=0.0,
            structured_outputs=StructuredOutputsParams(regex=r"[0-9]+"),
        ),
    )
    text = out[0].outputs[0].text.strip()
    assert text and all(c.isdigit() for c in text)


@pytest.mark.precommit
def test_custom_logits_processor(llm):
    """A plain request picks the token to ban; the same request with it banned must not emit it."""
    from vllm import SamplingParams

    baseline = llm.generate([{"prompt": PROMPT}], SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True))
    banned_token_id = baseline[0].outputs[0].token_ids[0]

    params = SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True,
                            extra_args={"ban_token_id": banned_token_id})
    banned = llm.generate([{"prompt": PROMPT}], params)[0].outputs[0]

    assert banned.token_ids[0] != banned_token_id


@pytest.mark.precommit
def test_varying_shapes_and_continuous_batching(llm):
    """Prefill lengths that change across generate() calls, then a mixed batch, on one model."""
    from vllm import SamplingParams

    long_prompt = "The quick brown fox jumps over the lazy dog near the riverbank at dawn."
    params = SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True)
    for prompts in (["Hi"], [long_prompt], ["Hi", long_prompt, "1 + 1 ="]):
        out = llm.generate([{"prompt": p} for p in prompts], params)
        assert len(out) == len(prompts)
        for prompt, o in zip(prompts, out):
            assert len(o.outputs[0].token_ids) == 8 and o.outputs[0].text, f"{prompt!r}: {o.outputs[0]}"


@pytest.mark.precommit
def test_beam_search(llm):
    """max_tokens=2: beam_search compiles and keeps a new OV model per decode step."""
    from vllm.sampling_params import BeamSearchParams

    out = llm.beam_search(
        [{"prompt": PROMPT}],
        BeamSearchParams(beam_width=3, max_tokens=2),
    )
    sequences = out[0].sequences
    assert len(sequences) == 3
    assert all(s.text for s in sequences)


@pytest.mark.precommit
def test_prefix_caching(llm):
    """Prefix caching is on by default; a full cache hit must reproduce the first output."""
    from vllm import SamplingParams

    shared_prefix = (
        "In a small village surrounded by mountains, there lived an old "
        "clockmaker who believed every gear told a story about the people "
        "who once needed it. "
    )
    params = SamplingParams(max_tokens=4, temperature=0.0, ignore_eos=True)

    text1 = llm.generate([{"prompt": shared_prefix + "One day,"}], params)[0].outputs[0].text
    text2 = llm.generate([{"prompt": shared_prefix + "One night,"}], params)[0].outputs[0].text
    text3 = llm.generate([{"prompt": shared_prefix + "One day,"}], params)[0].outputs[0].text

    assert text1.strip() and text2.strip()
    assert text1 == text3
