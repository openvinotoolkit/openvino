# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OV-fused sampler fast path for vLLM."""

import logging
import os

logger = logging.getLogger(__name__)


def _is_fastpath_eligible(sampling_metadata) -> bool:
    if sampling_metadata is None:
        return False
    if getattr(sampling_metadata, "max_num_logprobs", None):
        return False
    if getattr(sampling_metadata, "logprob_token_ids", None):
        return False
    procs = getattr(sampling_metadata, "logitsprocs", None)
    if procs is not None:
        # Allow processors that are no-ops on the current batch. The built-in
        # MinPLogitsProcessor is always present but short-circuits when no row
        # has min_p set (min_p_count == 0).
        def _is_noop(p):
            if hasattr(p, "min_p_count"):
                return p.min_p_count == 0
            if hasattr(p, "min_toks"):
                return not p.min_toks
            if hasattr(p, "biases"):
                return not p.biases
            return False
        for _proc in (getattr(procs, "argmax_invariant", None) or []):
            if not _is_noop(_proc):
                return False
        for _proc in (getattr(procs, "non_argmax_invariant", None) or []):
            if not _is_noop(_proc):
                return False
    if getattr(sampling_metadata, "all_greedy", False):
        return False
    if getattr(sampling_metadata, "temperature", None) is None:
        return False
    return True


_OV_SAMPLE_COMPILED = None


def _build_fused_sampler():
    import torch

    def _impl(logits, temperature, top_k, top_p, exp_noise):
        scaled = logits / temperature.unsqueeze(-1)
        max_k = top_k.max()
        V = logits.shape[-1]
        max_k_c = torch.clamp(max_k, max=V)
        topk_vals, topk_idx = torch.topk(scaled, k=max_k_c, dim=-1, sorted=True)
        col_idx = torch.arange(max_k_c, device=logits.device).unsqueeze(0)
        keep_k = col_idx < top_k.unsqueeze(-1)
        topk_softmax = torch.softmax(topk_vals, dim=-1)
        cumprob = torch.cumsum(topk_softmax, dim=-1)
        keep_p = (cumprob - topk_softmax) < top_p.unsqueeze(-1)
        keep_p = keep_p | (col_idx == 0)
        keep = keep_k & keep_p
        masked_vals = torch.where(keep, topk_vals, torch.full_like(topk_vals, -1e30))
        probs = torch.softmax(masked_vals, dim=-1)
        gathered_noise = torch.gather(exp_noise, dim=-1, index=topk_idx)
        scores = probs / gathered_noise
        winner = scores.argmax(dim=-1, keepdim=True)
        sampled = torch.gather(topk_idx, dim=-1, index=winner).squeeze(-1)
        return sampled.to(torch.int64)

    options = {"aot_autograd": True, "vllm": True}
    compiled = torch.compile(
        _impl,
        backend="openvino",
        fullgraph=False,
        dynamic=False,
        options=options,
    )
    return compiled


def install():
    if os.environ.get("OV_DISABLE_FUSED_SAMPLER"):
        return
    try:
        from vllm.v1.sample.sampler import Sampler
    except Exception as e:
        logger.debug("[OV plugin] Sampler import failed: %s", e)
        return
    if getattr(Sampler, "_ov_fused_installed", False):
        return

    _orig_sample = Sampler.sample

    def _patched_sample(self, logits, sampling_metadata, logprobs_mode_override=None):
        global _OV_SAMPLE_COMPILED
        if not _is_fastpath_eligible(sampling_metadata):
            return _orig_sample(self, logits, sampling_metadata,
                                logprobs_mode_override=logprobs_mode_override)
        if _OV_SAMPLE_COMPILED is None:
            try:
                _OV_SAMPLE_COMPILED = _build_fused_sampler()
                logger.info("[OV plugin] Fused sampler compiled")
            except Exception as e:
                logger.warning("[OV plugin] Fused sampler build failed: %s", e)
                Sampler.sample = _orig_sample
                return _orig_sample(self, logits, sampling_metadata,
                                    logprobs_mode_override=logprobs_mode_override)

        import torch
        logits_f32 = logits.to(torch.float32) if logits.dtype != torch.float32 else logits
        temperature = sampling_metadata.temperature
        if temperature.dim() == 0:
            temperature = temperature.unsqueeze(0).expand(logits_f32.shape[0])
        B, V = logits_f32.shape
        if sampling_metadata.top_k is not None:
            top_k = sampling_metadata.top_k.to(torch.int32)
        else:
            top_k = torch.full((B,), V, dtype=torch.int32)
        if sampling_metadata.top_p is not None:
            top_p = sampling_metadata.top_p.to(torch.float32)
        else:
            top_p = torch.full((B,), 1.0, dtype=torch.float32)

        exp_noise = torch.empty_like(logits_f32)
        gens = getattr(sampling_metadata, "generators", None) or {}
        if len(gens) != B:
            exp_noise.exponential_()
        if gens:
            for i, gen in gens.items():
                exp_noise[i].exponential_(generator=gen)

        try:
            sampled = _OV_SAMPLE_COMPILED(logits_f32, temperature, top_k, top_p, exp_noise)
        except Exception as e:
            logger.warning("[OV plugin] Fused sampler call failed, falling back: %s", e)
            return _orig_sample(self, logits, sampling_metadata,
                                logprobs_mode_override=logprobs_mode_override)

        return sampled, None

    Sampler.sample = _patched_sample
    Sampler._ov_fused_installed = True
    logger.info("[OV plugin] Sampler.sample patched")
