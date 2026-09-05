# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""vLLM general plugin for OpenVINO torch.compile backend.

Registered via the `vllm.general_plugins` setuptools entry point. vLLM calls
this at engine init and at worker startup. We use it to:

1. Patch `vllm.v1.worker.cpu_model_runner.CPUModelRunner.load_model` so it
   wires `torch.compile(backend="openvino")` when the user passes
   `compilation_config={"mode": "STOCK_TORCH_COMPILE", "backend": "openvino"}`.

2. Force vLLM's `_supports_onednn` to False when the openvino backend is in
   use, since onednn_mm graph-breaks the OV trace and rejects f32 activations
   from AOT decomposition.

3. Disable vLLM's LayerName opaque wrapper (env var equivalent of
   VLLM_USE_LAYERNAME=0) so OV's paged_attention C++ translator can cast
   the layer_name arg as `str`.

This module is a no-op on non-CPU vLLM installs and on any environment where
the OV backend is not requested by the user.
"""

import logging
import os

logger = logging.getLogger(__name__)


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

        # Install OV-fused sampler now that we know this worker is on OV.
        # Gated here (not in register()) so eager/inductor workers do not
        # get monkey-patched with the OV sampler path.
        try:
            from openvino.frontend.pytorch.torchdynamo.vllm import sampler as _vs
            _vs.install()
        except Exception as _e:
            logger.debug("[OV plugin] sampler install skipped: %s", _e)

        import torch
        try:
            import openvino.torch  # noqa: F401  (registers backend)
        except Exception as e:
            logger.warning("OV plugin: failed to import openvino.torch: %s", e)
            return

        # Widen the process CPU affinity if vLLM's init_cpu_threads_env
        # narrowed it to a single core. Must run before torch.compile so that
        # TBB/OV thread pools inherit the wide mask at creation.
        try:
            from openvino.frontend.pytorch.torchdynamo.vllm import compile_hooks as _vh_aff
            _vh_aff.widen_affinity_if_needed(None)
        except Exception as _e:
            logger.debug("[OV plugin] affinity widen skipped: %s", _e)

        logger.info("[OV plugin] Compiling model with torch.compile backend=openvino")
        # The "vllm": True mega-preset turns on every vLLM-required flag
        # (paged_attention, pa_translate, unbind_affinity, no_fallback,
        # fc_decompress) and seeds vLLM-specific OV config defaults
        # (KV_CACHE_PRECISION=bf16, INFERENCE_PRECISION_HINT=bf16,
        # DYNAMIC_QUANTIZATION_GROUP_SIZE=32). Individual flags can be
        # overridden by adding them explicitly to `options`.
        options = {"aot_autograd": True, "vllm": True}
        # dynamic=None (torch's default), not False: every distinct prefill
        # token count is a dynamo guard failure under False, costing a ~5.4 s
        # retrace plus a ~14 s OV compile_model, which recurs forever under a
        # varying request mix. None specializes the first shape, then
        # automatic_dynamic_shapes makes the varying dim symbolic on the 2nd
        # distinct length and stops retracing. Preferred over dynamic=True
        # because it marks only the dims that actually varied rather than all
        # of them: ~5% over the static graph in steady state vs True's ~1.7x.
        # See vllm/docs/dynamic_shapes.md for the measurements and for the
        # frontend/backend fixes the symbolic graph depends on.
        compiled = torch.compile(
            self.model.forward,
            backend="openvino",
            fullgraph=False,
            dynamic=None,
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


def _apply_default_env():
    """Set env-var defaults that make the OV backend perform well out of the
    box. Any user-set value takes priority; we only fill in unset vars."""
    defaults = {
        # vLLM's KV-cache pool size on CPU (GiB). Zero means "size to fill node
        # memory", which OOM-kills on shared NUMA nodes. 4 GiB is enough for
        # 1-2B models at 2k ctx; users can raise for larger models.
        "VLLM_CPU_KVCACHE_SPACE": "4",
        # Fast-path in torchdynamo/execute.py that caches set_tensor bindings
        # and output views across steps. +5-15% greedy on the models we test;
        # no correctness regression on Gemma-4 hybrid attention.
        "OV_FAST_INFER": "1",
    }
    for k, v in defaults.items():
        if os.environ.get(k) is None:
            os.environ[k] = v
            logger.debug("[OV plugin] %s=%s (default)", k, v)


def _warn_if_unpinned():
    """Warn if the process is not pinned to a subset of CPUs. Full-machine
    affinity often means threads land on multiple NUMA nodes, which halves
    memory bandwidth on decode. Users should launch with `taskset -c 0-N` or
    `numactl --cpunodebind=0` to pin to one socket.

    Best-effort only: skipped on platforms without sched_getaffinity (macOS,
    Windows) or when the affinity mask cannot be read.
    """
    try:
        allowed = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        return
    ncpus = os.cpu_count() or 0
    if ncpus and len(allowed) >= ncpus:
        logger.warning(
            "[OV plugin] process is not CPU-pinned (%d cores visible). "
            "For best decode throughput on multi-socket systems, launch with "
            "`taskset -c 0-N` or `numactl --cpunodebind=0 --membind=0` to pin "
            "to a single NUMA node.",
            len(allowed),
        )


def register():
    """Entry point for `vllm.general_plugins`.

    OV-active detection deferred to patched_load_model so we can read the
    real compilation_config.backend instead of guessing from env vars.
    """
    _apply_default_env()
    _disable_layername()
    _warn_if_unpinned()
    _patch_cpu_model_runner()
