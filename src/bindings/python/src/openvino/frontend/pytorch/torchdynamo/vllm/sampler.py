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
_OV_NATIVE_COMPILED = {}  # cached per (vocab, top_k, dtype)


def _build_native_sampler(vocab: int, k: int):
    """Build a native OpenVINO Model directly (opset13), skipping the
    torch.compile trace + dynamo overhead.

    Graph: logits[B,V] + temperature[B] -> sampled_token[B]
      1. divide logits by temperature (broadcast)
      2. topk(k) -> values[B,k], indices[B,k]
      3. softmax(values)
      4. RandomUniform + log(-log()) = Gumbel noise on [B,k]
      5. score = log(probs) + gumbel
      6. argmax over k, gather winner from indices
    """
    import numpy as np
    import openvino as ov
    from openvino import opset13 as op
    from openvino import Type, PartialShape, Model, Core

    logits_p = op.parameter(PartialShape([-1, vocab]), Type.f32, name="logits")
    temp_p = op.parameter(PartialShape([-1]), Type.f32, name="temperature")
    temp_exp = op.unsqueeze(temp_p, op.constant(-1, Type.i32))
    scaled = op.divide(logits_p, temp_exp)
    tk = op.topk(scaled, k=op.constant(k, Type.i32), axis=-1, mode="max", sort="value")
    tv, ti = tk.output(0), tk.output(1)
    probs = op.softmax(tv, axis=-1)
    shape_of = op.shape_of(tv, Type.i64)
    u = op.random_uniform(shape_of,
                           op.constant(0.0, Type.f32),
                           op.constant(1.0, Type.f32),
                           Type.f32, global_seed=0, op_seed=0)
    eps = op.constant(1e-20, Type.f32)
    neg = op.constant(-1.0, Type.f32)
    g = op.multiply(op.log(op.add(op.multiply(op.log(op.add(u, eps)), neg), eps)), neg)
    score = op.add(op.log(op.add(probs, op.constant(1e-30, Type.f32))), g)
    idx = op.topk(score, k=op.constant(1, Type.i32), axis=-1, mode="max", sort="value").output(1)
    token = op.gather_elements(ti, idx, axis=-1)
    token_i32 = op.convert(token, Type.i32)
    model = Model([token_i32.output(0)], [logits_p, temp_p], f"ov_native_sampler_v{vocab}_k{k}")

    core = Core()
    cfg = {
        "INFERENCE_PRECISION_HINT": os.environ.get("OV_FAST_SAMPLER_HINT", "f32"),
        "INFERENCE_NUM_THREADS": int(os.environ.get("OV_INFERENCE_NUM_THREADS", "40")),
    }
    compiled = core.compile_model(model, "CPU", cfg)
    return compiled


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
        # Gate on vocab size: below ~100k, torch's topk_topp_sampler on CPU
        # is faster than round-tripping through a compiled OV graph. The
        # threshold is tunable via OV_FUSED_SAMPLER_MIN_VOCAB. Also skip when
        # logits shape[0] > 1 (batching path is untested).
        _min_vocab = int(os.environ.get("OV_FUSED_SAMPLER_MIN_VOCAB", "100000"))
        if logits.shape[-1] < _min_vocab or logits.shape[0] > 1:
            return _orig_sample(self, logits, sampling_metadata,
                                logprobs_mode_override=logprobs_mode_override)
        import torch
        _use_native = os.environ.get("OV_NATIVE_SAMPLER", "0") != "0"
        B, V = logits.shape

        if _use_native:
            # Native OV graph path — no torch.compile overhead. Skips top_p
            # (uses pure Gumbel-max over top_k values) and has no per-request
            # seed. Users opt in via OV_NATIVE_SAMPLER=1 and accept the small
            # distribution difference when top_p < 1.0.
            top_k_meta = getattr(sampling_metadata, "top_k", None)
            try:
                k_val = int(top_k_meta.max().item()) if top_k_meta is not None else 0
            except Exception:
                k_val = 0
            if k_val <= 0 or k_val > 128:
                _use_native = False

            if _use_native:
                cache_key = (V, k_val)
                compiled = _OV_NATIVE_COMPILED.get(cache_key)
                if compiled is None:
                    try:
                        compiled = _build_native_sampler(V, k_val)
                        _OV_NATIVE_COMPILED[cache_key] = compiled
                        logger.info(f"[OV plugin] Native sampler compiled (V={V}, k={k_val})")
                    except Exception as e:
                        logger.warning("[OV plugin] Native sampler build failed: %s", e)
                        _use_native = False

            if _use_native:
                import numpy as np
                logits_f32 = logits.to(torch.float32) if logits.dtype != torch.float32 else logits
                temperature = sampling_metadata.temperature
                if temperature.dim() == 0:
                    temperature = temperature.unsqueeze(0).expand(B)
                temp_f32 = temperature.to(torch.float32) if temperature.dtype != torch.float32 else temperature
                try:
                    l_np = logits_f32.detach().cpu().numpy()
                    t_np = temp_f32.detach().cpu().numpy()
                    res = compiled([l_np, t_np])
                    tok = list(res.values())[0]
                    sampled = torch.from_numpy(tok.reshape(-1)).to(torch.int64)
                    return sampled, None
                except Exception as e:
                    logger.warning("[OV plugin] Native sampler call failed, falling back: %s", e)
                    _use_native = False

        # Torch-compiled fused sampler path (original behavior)
        if _OV_SAMPLE_COMPILED is None:
            try:
                _OV_SAMPLE_COMPILED = _build_fused_sampler()
                logger.info("[OV plugin] Fused sampler compiled")
            except Exception as e:
                logger.warning("[OV plugin] Fused sampler build failed: %s", e)
                Sampler.sample = _orig_sample
                return _orig_sample(self, logits, sampling_metadata,
                                    logprobs_mode_override=logprobs_mode_override)

        logits_f32 = logits.to(torch.float32) if logits.dtype != torch.float32 else logits
        temperature = sampling_metadata.temperature
        if temperature.dim() == 0:
            temperature = temperature.unsqueeze(0).expand(logits_f32.shape[0])
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
