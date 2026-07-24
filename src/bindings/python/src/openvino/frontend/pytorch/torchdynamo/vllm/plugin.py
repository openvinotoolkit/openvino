# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""vLLM general plugin for OpenVINO torch.compile backend.

Registered via the `vllm.general_plugins` setuptools entry point. vLLM calls
this at engine init and at worker startup. We use it to:

1. Patch `vllm.v1.worker.cpu_model_runner.CPUModelRunner.load_model` so it
   wires `torch.compile(backend="openvino")` when the user passes
   `compilation_config={"mode": "STOCK_TORCH_COMPILE", "backend": "openvino"}`.
   Replaces the user-visible Patch C from spr_perf_results.md.

2. Force vLLM's `_supports_onednn` to False when the openvino backend is in
   use, since onednn_mm graph-breaks the OV trace and rejects f32 activations
   from AOT decomposition. Replaces Patch D.

3. Disable vLLM's LayerName opaque wrapper (env var equivalent of
   VLLM_USE_LAYERNAME=0) so OV's paged_attention C++ translator can cast
   the layer_name arg as `str`.

This module is no-op on non-CPU vLLM installs and on any environment where
the OV backend is not requested by the user.
"""

import logging
import os

logger = logging.getLogger(__name__)


def _is_openvino_requested():
    """True when user wants the OV backend, before vLLM has constructed config."""
    # Cheap heuristic: rely on env vars users typically set together with
    # the openvino backend. Refined at patch-time below.
    return True  # always patch — checks happen inside patched load_model


def _patch_cpu_model_runner():
    try:
        from vllm.v1.worker.cpu_model_runner import CPUModelRunner
    except Exception:
        return

    if getattr(CPUModelRunner, "_ov_plugin_patched", False):
        return

    _orig_load_model = CPUModelRunner.load_model

    def _ov_active(self) -> bool:
        comp_cfg = getattr(self.vllm_config, "compilation_config", None)
        if comp_cfg is None:
            return False
        try:
            mode = getattr(comp_cfg, "mode", None)
            mode_name = getattr(mode, "name", None) if mode is not None else None
            backend = getattr(comp_cfg, "backend", None)
        except Exception:
            return False
        return mode_name == "STOCK_TORCH_COMPILE" and backend == "openvino"

    def patched_load_model(self, load_dummy_weights: bool = False) -> None:
        # Resolve OV-active state from the actual vllm_config (not env vars).
        # Flip _supports_onednn BEFORE _orig_load_model so vLLM's FC layers
        # see the False value when they're constructed during model load.
        is_ov = _ov_active(self)
        if is_ov:
            try:
                import vllm._custom_ops as _ops
                if not getattr(_ops, "_ov_plugin_onednn_disabled", False):
                    _ops._supports_onednn = False
                    _ops._ov_plugin_onednn_disabled = True
                    logger.debug("[OV plugin] _supports_onednn forced False (backend=openvino)")
            except Exception as _e:
                logger.debug("[OV plugin] _supports_onednn flip skipped: %s", _e)

        _orig_load_model(self, load_dummy_weights)
        if not is_ov:
            return

        import torch
        try:
            import openvino.torch  # noqa: F401  (registers backend)
        except Exception as e:
            logger.warning("OV plugin: failed to import openvino.torch: %s", e)
            return

        logger.info("[OV plugin] Compiling model with torch.compile backend=openvino")
        # The "vllm": True mega-preset turns on every vLLM-required flag
        # (paged_attention, pa_translate, unbind_affinity, no_fallback,
        # fc_decompress) and seeds vLLM-specific OV config defaults
        # (KV_CACHE_PRECISION=bf16, INFERENCE_PRECISION_HINT=bf16,
        # DYNAMIC_QUANTIZATION_GROUP_SIZE=32). Individual flags can be
        # overridden by adding them explicitly to `options`.
        options = {"aot_autograd": False, "vllm": True} if os.environ.get("OV_NO_AOT") else {"aot_autograd": True, "vllm": True}
        if os.environ.get("OV_PA_TORCH_FALLBACK"): options["pa_translate"] = False
        compiled = torch.compile(
            self.model.forward,
            backend="openvino",
            fullgraph=False,
            dynamic=False,
            options=options,
        )
        self.model.forward = compiled

        # lm_head runs OUTSIDE the OV-compiled forward() (in compute_logits()).
        # It got cpu_linear=torch.nn.functional.linear during model load
        # because onednn was disabled to keep OV trace clean. Re-dispatch
        # just lm_head with onednn enabled so its huge [hidden, vocab]
        # bf16 GEMM uses oneDNN's AMX-prepacked path. Saves ~3-5 ms/step
        # at Llama-3.2-1B decode (lm_head reads 524 MB weight per call).
        try:
            import vllm._custom_ops as _ops
            from vllm.model_executor.layers.utils import dispatch_cpu_unquantized_gemm
            lm_head = getattr(self.model, "lm_head", None)
            if lm_head is not None and hasattr(lm_head, "weight") and not lm_head.weight.is_meta:
                _saved = getattr(_ops, "_supports_onednn", True)
                _ops._supports_onednn = True
                try:
                    dispatch_cpu_unquantized_gemm(lm_head, remove_weight=False)
                    logger.info("[OV plugin] lm_head re-dispatched with onednn enabled")
                except Exception as _e:
                    logger.warning("[OV plugin] lm_head onednn re-dispatch failed: %s", _e)
                finally:
                    _ops._supports_onednn = _saved
        except Exception as _e:
            logger.debug("[OV plugin] lm_head onednn fast path unavailable: %s", _e)

    CPUModelRunner.load_model = patched_load_model
    CPUModelRunner._ov_plugin_patched = True
    logger.debug("[OV plugin] CPUModelRunner.load_model patched")


def _disable_layername():
    """Force VLLM_USE_LAYERNAME=0 so OV PA op gets plain str layer names."""
    if os.environ.get("VLLM_USE_LAYERNAME") is None:
        os.environ["VLLM_USE_LAYERNAME"] = "0"
        logger.debug("[OV plugin] VLLM_USE_LAYERNAME=0 set")


def register():
    """Entry point for `vllm.general_plugins`.

    OV-active detection deferred to patched_load_model so we can read the
    real compilation_config.backend instead of guessing from env vars.
    """
    _disable_layername()
    _patch_cpu_model_runner()
    try:
        from openvino.frontend.pytorch.torchdynamo.vllm import sampler as _vs
        _vs.install()
    except Exception as _e:
        logger.debug("[OV plugin] sampler install skipped: %s", _e)
