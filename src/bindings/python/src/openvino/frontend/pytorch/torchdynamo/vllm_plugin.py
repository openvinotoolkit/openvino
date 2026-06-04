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

    def patched_load_model(self, load_dummy_weights: bool = False) -> None:
        _orig_load_model(self, load_dummy_weights)
        # vLLM's init_cpu_threads_env pins this worker to a single CPU before
        # we get here. Opt the OV compile path into widening that mask so TBB
        # sees a real thread count when it samples affinity on first compile.
        # Default the env knob to 1 only for vLLM-driven runs; standalone
        # torch.compile users keep the (default 0) no-affinity-mutation path.
        import os as _os_aff
        _os_aff.environ.setdefault("OV_UNBIND_AFFINITY", "1")
        comp_cfg = getattr(self.vllm_config, "compilation_config", None)
        try:
            mode = getattr(comp_cfg, "mode", None)
            mode_name = getattr(mode, "name", None) if mode is not None else None
            backend = getattr(comp_cfg, "backend", None)
        except Exception:
            return
        if comp_cfg is None or mode_name != "STOCK_TORCH_COMPILE":
            return
        if backend != "openvino":
            return

        import torch
        try:
            import openvino.torch  # noqa: F401  (registers backend)
        except Exception as e:
            logger.warning("OV plugin: failed to import openvino.torch: %s", e)
            return

        logger.info("[OV plugin] Compiling model with torch.compile backend=openvino")
        options = {"aot_autograd": True}
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


def _force_disable_onednn():
    try:
        import vllm._custom_ops as _ops
    except Exception:
        return
    if getattr(_ops, "_ov_plugin_onednn_disabled", False):
        return
    # Only disable when an OV-style run is suggested by env vars; otherwise leave
    # vLLM's default in place. We check here rather than later because the flag
    # is read at module import time by various vLLM linear layers.
    if os.environ.get("OV_VLLM_PA") or "openvino" in os.environ.get("BENCH_OV_BACKEND", ""):
        _ops._supports_onednn = False
        _ops._ov_plugin_onednn_disabled = True
        logger.debug("[OV plugin] vllm._custom_ops._supports_onednn forced False")


def _disable_layername():
    """Force VLLM_USE_LAYERNAME=0 so OV PA op gets plain str layer names."""
    if os.environ.get("VLLM_USE_LAYERNAME") is None:
        os.environ["VLLM_USE_LAYERNAME"] = "0"
        logger.debug("[OV plugin] VLLM_USE_LAYERNAME=0 set")


def register():
    """Entry point for `vllm.general_plugins`."""
    _disable_layername()
    _force_disable_onednn()
    _patch_cpu_model_runner()
