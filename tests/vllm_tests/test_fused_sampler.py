# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Coverage for the OV-fused stochastic sampler (sampler.py), which no generation test reaches.

The generation tests run TinyLlama (32000-token vocab, below the default
OV_FUSED_SAMPLER_MIN_VOCAB=100000 gate) at temperature=0 (all_greedy, also
ineligible), so the fused top-k/top-p + seeded-noise path never runs there.
These tests drive Sampler.sample directly on random logits with the gate
lowered, and compare the fused result token-for-token against vLLM's own
CPU seeded-sampling recipe (apply_top_k_top_p + softmax + exponential
noise drawn from the request generator over the full vocab). The fused
path draws the same noise from the same generator, so with a seed the
two must pick the same token -- not just a statistically equivalent one.

No model download needed.
"""

import pytest
import torch

from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
from vllm.v1.sample.sampler import Sampler

from openvino.frontend.pytorch.torchdynamo.vllm import sampler as ov_sampler

VOCAB = 32000

CASES = [
    # temperature, top_k, top_p
    (0.7, 50, 0.9),
    (1.0, None, 0.8),
    (1.3, 20, None),
    (0.5, 1000, 0.95),
    (2.0, None, 0.5),
    (1.0, None, None),
    (0.8, 40, 0.3),
]


def _pristine_sample():
    """vLLM's own Sampler.sample, unwrapping the OV patch if register() already ran in this process."""
    fn = Sampler.sample
    while getattr(fn, "__name__", "") == "_patched_sample":
        cells = dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in fn.__closure__)))
        fn = cells["_orig_sample"]
    return fn


def _metadata(temperature, top_k, top_p, seed=None, max_num_logprobs=None):
    generators = {}
    if seed is not None:
        generators[0] = torch.Generator().manual_seed(seed)
    return SamplingMetadata(
        temperature=torch.tensor([temperature], dtype=torch.float32),
        all_greedy=False,
        all_random=True,
        top_p=None if top_p is None else torch.tensor([top_p], dtype=torch.float32),
        top_k=None if top_k is None else torch.tensor([top_k], dtype=torch.int32),
        generators=generators,
        max_num_logprobs=max_num_logprobs,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(1),
        presence_penalties=torch.zeros(1),
        repetition_penalties=torch.ones(1),
        output_token_ids=[[]],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def _reference(logits, temperature, top_k, top_p, seed):
    """vLLM TopKTopPSampler.forward_cpu for one seeded request, plus the kept-token mask."""
    scaled = logits.clone() / temperature
    filtered = apply_top_k_top_p_pytorch(
        scaled,
        None if top_k is None else torch.tensor([top_k]),
        None if top_p is None else torch.tensor([top_p]),
    )
    probs = filtered.softmax(dim=-1, dtype=torch.float32)
    noise = torch.empty_like(probs)
    noise[0].exponential_(generator=torch.Generator().manual_seed(seed))
    return int(probs.div(noise).argmax(dim=-1)), torch.isfinite(filtered[0])


@pytest.fixture
def fused_sampler(monkeypatch):
    """Install the OV sampler patch over a fallback that records every call.

    Restores Sampler.sample and the compiled-sampler cache afterwards so no
    other test in the session sees this test's patch.
    """
    monkeypatch.setenv("OV_FUSED_SAMPLER_MIN_VOCAB", "1")
    monkeypatch.delenv("OV_DISABLE_FUSED_SAMPLER", raising=False)
    monkeypatch.delenv("OV_NATIVE_SAMPLER", raising=False)
    pristine = _pristine_sample()
    fallbacks = []

    def _recording_fallback(self, logits, sampling_metadata, logprobs_mode_override=None):
        fallbacks.append(sampling_metadata)
        return pristine(self, logits, sampling_metadata, logprobs_mode_override=logprobs_mode_override)

    monkeypatch.setattr(Sampler, "sample", _recording_fallback)
    monkeypatch.setattr(Sampler, "_ov_fused_installed", False, raising=False)
    monkeypatch.setattr(ov_sampler, "_OV_SAMPLE_COMPILED", None)
    ov_sampler.install()
    assert Sampler.sample is not _recording_fallback, "sampler.install() did not patch Sampler.sample"
    return fallbacks


@pytest.mark.precommit
@pytest.mark.parametrize("temperature,top_k,top_p", CASES)
def test_fused_sampler_matches_vllm_seeded(fused_sampler, temperature, top_k, top_p):
    torch.manual_seed(0)
    sampler = Sampler()
    for seed in range(10):
        logits = torch.randn(1, VOCAB) * 4.0
        expected, kept = _reference(logits, temperature, top_k, top_p, seed)
        got, _ = sampler.sample(logits.clone(), _metadata(temperature, top_k, top_p, seed))
        again, _ = sampler.sample(logits.clone(), _metadata(temperature, top_k, top_p, seed))
        assert not fused_sampler, "fused sampler fell back to vLLM's sampler"
        token = int(got[0])
        assert kept[token], f"seed {seed}: token {token} is outside the top-k/top-p set"
        assert token == int(again[0]), f"seed {seed}: same seed sampled a different token"
        assert token == expected, f"seed {seed}: fused sampled {token}, vLLM sampled {expected}"


@pytest.mark.precommit
@pytest.mark.parametrize("allowed", [1, 3, 200])
def test_fused_sampler_masked_logits(fused_sampler, allowed):
    """-inf logits (allowed_token_ids / bad_words, applied before sample) must never be sampled."""
    torch.manual_seed(1)
    sampler = Sampler()
    for seed in range(5):
        logits = torch.randn(1, VOCAB) * 4.0
        keep = torch.randperm(VOCAB)[:allowed]
        masked = torch.full_like(logits, float("-inf"))
        masked[0, keep] = logits[0, keep]
        expected, _ = _reference(masked, 0.9, 50, 0.9, seed)
        got, _ = sampler.sample(masked.clone(), _metadata(0.9, 50, 0.9, seed))
        assert not fused_sampler, "fused sampler fell back to vLLM's sampler"
        assert int(got[0]) in keep.tolist()
        assert int(got[0]) == expected


@pytest.mark.precommit
@pytest.mark.parametrize("logprobs_mode", ["raw_logprobs", "processed_logprobs"])
def test_fused_sampler_logprobs_zero(fused_sampler, logprobs_mode):
    """logprobs=0 (max_num_logprobs == 0) still needs the sampled token's logprob.

    In processed_* modes that logprob comes from sample()'s second return
    value, which the fused path always sets to None.
    """
    sampler = Sampler(logprobs_mode=logprobs_mode)
    logits = torch.randn(1, VOCAB) * 4.0
    out = sampler(logits, _metadata(1.0, 50, 0.9, seed=3, max_num_logprobs=0))
    assert out.logprobs_tensors is not None
    assert torch.isfinite(out.logprobs_tensors.logprobs[:, 0]).all()
