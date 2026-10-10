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

from conftest import new_openvino_llm

PROMPTS = [
    "Hi",
    ("In a small village surrounded by mountains, there lived an old "
     "clockmaker who believed every gear told a story about the people "
     "who once needed it."),
]


@pytest.mark.precommit
def test_varying_sequence_lengths_reuse_compiled_cache():
    """Same loaded model, prefill lengths short then long: both calls must succeed.

    Own LLM instance: cache reuse is within-instance, and this compiles a
    second shape (see new_openvino_llm).
    """
    from vllm import SamplingParams

    llm = new_openvino_llm()
    try:
        for prompt in PROMPTS:
            params = SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True)
            out = llm.generate([{"prompt": prompt}], params)
            result = out[0].outputs[0]

            assert len(result.token_ids) == 8, (
                f"prompt {prompt!r} produced {len(result.token_ids)} tokens, expected 8 "
                "-- looks like the shape-keyed cache served a stale/wrong-shaped output")
            assert result.text, f"prompt {prompt!r} produced empty output text"
    finally:
        del llm
