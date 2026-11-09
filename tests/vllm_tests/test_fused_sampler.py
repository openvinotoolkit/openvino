# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The OV-fused seeded sampler must pick exactly the token vLLM's own CPU recipe picks.

No generation test reaches it (TinyLlama is below the vocab gate and decodes greedily), so these call it directly.
"""

import pytest
import torch

from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
from vllm.v1.sample.sampler import Sampler

from openvino.frontend.pytorch.torchdynamo.vllm import sampler as ov_sampler

VOCAB = 32000

# temperature, top_k, top_p: both filters, top-p only, top-k only, neither
CASES = [(0.7, 50, 0.9), (1.0, None, 0.8), (1.3, 20, None), (1.0, None, None)]


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


def _install(monkeypatch, fallback):
    """Install the OV sampler patch over `fallback`; monkeypatch undoes it for later tests."""
    monkeypatch.setenv("OV_FUSED_SAMPLER_MIN_VOCAB", "1")
    monkeypatch.delenv("OV_DISABLE_FUSED_SAMPLER", raising=False)
    monkeypatch.delenv("OV_NATIVE_SAMPLER", raising=False)
    monkeypatch.setattr(Sampler, "sample", fallback)
    monkeypatch.setattr(Sampler, "_ov_fused_installed", False, raising=False)
    monkeypatch.setattr(ov_sampler, "_OV_SAMPLE_COMPILED", None)
    ov_sampler.install()
    assert Sampler.sample is not fallback, "sampler.install() did not patch Sampler.sample"


@pytest.fixture
def fused_sampler(monkeypatch):
    """The fallback calls vLLM's sampler and records each call."""
    pristine = _pristine_sample()
    fallbacks = []

    def _recording_fallback(self, logits, sampling_metadata, logprobs_mode_override=None):
        fallbacks.append(sampling_metadata)
        return pristine(self, logits, sampling_metadata, logprobs_mode_override=logprobs_mode_override)

    _install(monkeypatch, _recording_fallback)
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
@pytest.mark.parametrize("far_negative", [False, True], ids=["random", "far_negative"])
def test_fused_sampler_masked_logits(fused_sampler, allowed, far_negative):
    """-inf logits (allowed_token_ids / bad_words) are never sampled, even when the allowed ones sit below -1e30."""
    torch.manual_seed(1)
    sampler = Sampler()
    for seed in range(5):
        keep = torch.randperm(VOCAB)[:allowed]
        masked = torch.full((1, VOCAB), float("-inf"))
        masked[0, keep] = (-1e31 * (1.0 + 0.1 * torch.arange(allowed, dtype=torch.float32)) if far_negative
                           else torch.randn(allowed) * 4.0)
        expected, _ = _reference(masked, 0.9, 50, 0.9, seed)
        got, _ = sampler.sample(masked.clone(), _metadata(0.9, 50, 0.9, seed))
        assert not fused_sampler, "fused sampler fell back to vLLM's sampler"
        assert int(got[0]) in keep.tolist()
        assert int(got[0]) == expected


@pytest.mark.precommit
@pytest.mark.parametrize("logprobs_mode", ["raw_logprobs", "processed_logprobs"])
def test_fused_sampler_logprobs_zero(fused_sampler, logprobs_mode):
    """logprobs=0 still needs the sampled token's logprob, which processed_* modes take from sample()'s output."""
    sampler = Sampler(logprobs_mode=logprobs_mode)
    logits = torch.randn(1, VOCAB) * 4.0
    out = sampler(logits, _metadata(1.0, 50, 0.9, seed=3, max_num_logprobs=0))
    assert out.logprobs_tensors is not None
    assert torch.isfinite(out.logprobs_tensors.logprobs[:, 0]).all()


@pytest.mark.precommit
def test_fused_sampler_keeps_topk_ties(fused_sampler):
    """Logits tied with the k-th largest all stay samplable, as in vLLM's top-k (topk alone keeps only some)."""
    sampler = Sampler()
    tied = [11, 22, 33, 44]
    logits = torch.full((1, VOCAB), -10.0)
    logits[0, 5] = 3.0
    logits[0, tied] = 2.0
    seen = set()
    for seed in range(40):
        expected, _ = _reference(logits, 1.0, 2, None, seed)
        got, _ = sampler.sample(logits.clone(), _metadata(1.0, 2, None, seed))
        assert not fused_sampler, "fused sampler fell back to vLLM's sampler"
        assert int(got[0]) == expected, f"seed {seed}: fused sampled {int(got[0])}, vLLM sampled {expected}"
        seen.add(int(got[0]))
    assert set(tied) <= seen, f"tied tokens never sampled: {sorted(set(tied) - seen)}"


@pytest.mark.precommit
def test_fused_sampler_tie_overflow_falls_back(fused_sampler):
    """More k-th ties than the fused top-k window holds: hand the step to vLLM rather than drop some."""
    sampler = Sampler()
    logits = torch.full((1, VOCAB), -10.0)
    logits[0, 5] = 3.0
    logits[0, 100:400] = 2.0
    for seed in range(3):
        expected, _ = _reference(logits, 1.0, 2, None, seed)
        got, _ = sampler.sample(logits.clone(), _metadata(1.0, 2, None, seed))
        assert int(got[0]) == expected
    assert len(fused_sampler) == 3, "tie overflow did not fall back"


@pytest.mark.precommit
def test_fused_sampler_fp64_gumbel_falls_back(fused_sampler):
    """use_fp64_gumbel asks for float64 noise, which the fused path doesn't draw."""
    sampler = Sampler(use_fp64_gumbel=True)
    sampler.sample(torch.randn(1, VOCAB), _metadata(0.7, 50, 0.9, seed=0))
    assert len(fused_sampler) == 1, "fused sampler ignored use_fp64_gumbel"


@pytest.mark.precommit
def test_fused_sampler_failure_restores_default_rng(monkeypatch):
    """A failed fused call must leave the default CPU RNG as it found it, so the fallback draws what vLLM would."""
    states = []

    def _fallback(self, logits, sampling_metadata, logprobs_mode_override=None):
        states.append(torch.get_rng_state())
        return torch.zeros(1, dtype=torch.int64), None

    def _broken(*args):
        raise RuntimeError("injected fused sampler failure")

    _install(monkeypatch, _fallback)
    monkeypatch.setattr(ov_sampler, "_OV_SAMPLE_COMPILED", _broken)
    torch.manual_seed(123)
    logits = torch.randn(1, VOCAB)
    expected_state = torch.get_rng_state()
    Sampler().sample(logits, _metadata(0.7, 50, 0.9))
    assert len(states) == 1, "the injected failure did not reach the fallback"
    assert torch.equal(states[0], expected_state), "fused attempt advanced the default RNG before the fallback"
