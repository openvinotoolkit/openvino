# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""n-gram speculative decoding through OV; in its own module, so its LLM is never alive next to the shared one."""

import pytest

from conftest import new_openvino_llm


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
