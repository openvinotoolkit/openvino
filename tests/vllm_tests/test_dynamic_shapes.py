# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for shape changes across generate() calls on one model.

torchdynamo/execute.py keys its compiled-model/request caches by the FX
partition's structural+shape signature (see execute.py's `structural_cache`
/ `_structural_key`). An earlier version keyed only by `partition_id`, so a
second call with a different prompt/sequence length reused the first call's
compiled entry and produced zero-sized outputs instead of recompiling or
selecting the right cached variant. This drives one already-loaded OV model
through several different prefill lengths to catch that class of bug.
"""

import pytest

PROMPTS = [
    "Hi",
    "The quick brown fox jumps over the lazy dog near the riverbank at dawn.",
    "1 + 1 =",
    ("In a small village surrounded by mountains, there lived an old "
     "clockmaker who believed every gear told a story about the people "
     "who once needed it."),
    "Hi",
]


@pytest.mark.precommit
def test_varying_sequence_lengths_reuse_compiled_cache(openvino_llm):
    """Same loaded model, prefill lengths short/long/short: every call must succeed."""
    from vllm import SamplingParams

    for prompt in PROMPTS:
        params = SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True)
        out = openvino_llm.generate([{"prompt": prompt}], params)
        result = out[0].outputs[0]

        assert len(result.token_ids) == 8, (
            f"prompt {prompt!r} produced {len(result.token_ids)} tokens, expected 8 "
            "-- looks like the shape-keyed cache served a stale/wrong-shaped output")
        assert result.text, f"prompt {prompt!r} produced empty output text"
