# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""vLLM general plugin for the OpenVINO torch.compile backend.

Registered via the `vllm.general_plugins` entry point. Patches
`CPUModelRunner.load_model` to wire `torch.compile(backend="openvino")`, plus
`reload_weights`/`CPUWorker.update_weights` to rebuild the OV lm_head after a
post-load weight mutation. No-op unless the user requested the OV backend.
"""

import logging
import os

logger = logging.getLogger(__name__)


def _ov_active(self) -> bool:
    """True if `self.vllm_config.compilation_config` requests the OV backend.

    Shared by the model-runner and worker patches below -- both expose
    `self.vllm_config`.
    """
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


def _install_ov_lm_head(model) -> None:
    """(Re)compile lm_head on OV and swap it onto `model.lm_head.cpu_linear`.

    Rerun on every weight mutation (reload_weights/update_weights below) since
    build_ov_lm_head bakes the weight into an OV constant. Defaults to the
    oneDNN dispatch (OV_LM_HEAD=0): the OV path roughly doubles main-graph
    per-step time (see lm_head.py). Set OV_LM_HEAD=1 to opt in.
    """
    try:
        import vllm._custom_ops as _ops
        from vllm.model_executor.layers.utils import dispatch_cpu_unquantized_gemm
        from openvino.frontend.pytorch.torchdynamo.vllm.lm_head import build_ov_lm_head
        lm_head = getattr(model, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "weight") and not lm_head.weight.is_meta:
            fn = None
            if os.environ.get("OV_LM_HEAD", "0") != "0":
                try:
                    fn = build_ov_lm_head(lm_head.weight.data)
                except Exception as _e:
                    logger.warning("[OV plugin] lm_head OV compile failed: %s", _e)
            if fn is not None:
                lm_head.cpu_linear = fn
            else:
                _saved = getattr(_ops, "_supports_onednn", True)
                _ops._supports_onednn = True
                try:
                    # remove_weight=False: create_onednn_mm consumes the
                    # weight, and the OV path may still need it.
                    dispatch_cpu_unquantized_gemm(lm_head, remove_weight=False)
                    logger.info("[OV plugin] lm_head re-dispatched with onednn enabled")
                except Exception as _e:
                    logger.warning("[OV plugin] lm_head onednn re-dispatch failed: %s", _e)
                finally:
                    _ops._supports_onednn = _saved
    except Exception as _e:
        logger.debug("[OV plugin] lm_head fast path unavailable: %s", _e)


_KV_UPDATE_BACKENDS: dict = {}


def _kv_update_owned_backend(base):
    """Memoized `base` subclass reporting forward_includes_kv_cache_update=True.

    Memoized because the runner dedupes attention groups by backend class
    (`full_cls_name`, gpu_model_runner.py:6997) and also collects the classes
    themselves into a set: handing out a freshly built class per layer would
    split the one CPU_ATTN group into N.
    """
    cached = _KV_UPDATE_BACKENDS.get(base)
    if cached is None:
        from vllm.v1.attention.backend import subclass_attention_backend_with_overrides
        cached = subclass_attention_backend_with_overrides(
            name_prefix="OVKVUpdate",
            attention_backend_cls=base,
            overrides={"forward_includes_kv_cache_update": True},
        )
        _KV_UPDATE_BACKENDS[base] = cached
    return cached


def _own_kv_cache_update(model) -> int:
    """Point this model's attention layers at a kv-update-owning backend.

    OV's PagedAttention does the paged write itself, so vLLM's separate
    unified_kv_cache_update op is redundant and, having
    no OV translator, would be stranded in eager PyTorch by the partitioner.
    Scoped per layer, not on CPUAttentionBackend, which is process-global.
    """
    from vllm.v1.attention.backend import AttentionBackend
    swapped = 0
    for module in model.modules():
        # Some layers hold an AttentionBackendEnum here, not a backend class.
        backend = getattr(module, "attn_backend", None)
        if (isinstance(backend, type) and issubclass(backend, AttentionBackend)
                and not backend.forward_includes_kv_cache_update):
            module.attn_backend = _kv_update_owned_backend(backend)
            swapped += 1
    return swapped


def _patch_cpu_model_runner():
    try:
        from vllm.v1.worker.cpu_model_runner import CPUModelRunner
    except Exception:
        return

    if getattr(CPUModelRunner, "_ov_plugin_patched", False):
        return

    _orig_load_model = CPUModelRunner.load_model

    def patched_load_model(self, load_dummy_weights: bool = False) -> None:
        # Flip _supports_onednn before load so vLLM's FC layers see it built.
        # else restores it for a later non-OV model in the same process.
        is_ov = _ov_active(self)
        if is_ov:
            try:
                import vllm._custom_ops as _ops
                if not getattr(_ops, "_ov_plugin_onednn_disabled", False):
                    _ops._ov_plugin_onednn_saved = getattr(_ops, "_supports_onednn", True)
                    _ops._supports_onednn = False
                    _ops._ov_plugin_onednn_disabled = True
                    logger.debug("[OV plugin] _supports_onednn forced False (backend=openvino)")
            except Exception as _e:
                logger.debug("[OV plugin] _supports_onednn flip skipped: %s", _e)
        else:
            try:
                import vllm._custom_ops as _ops
                if getattr(_ops, "_ov_plugin_onednn_disabled", False):
                    _ops._supports_onednn = getattr(_ops, "_ov_plugin_onednn_saved", True)
                    _ops._ov_plugin_onednn_disabled = False
                    logger.debug("[OV plugin] _supports_onednn restored (backend != openvino)")
            except Exception as _e:
                logger.debug("[OV plugin] _supports_onednn restore skipped: %s", _e)

        _orig_load_model(self, load_dummy_weights)
        if not is_ov:
            return

        # Hand the KV-cache write to OV's PagedAttention for this model's
        # layers only; see _own_kv_cache_update.
        try:
            _n = _own_kv_cache_update(self.model)
            logger.debug(
                "[OV plugin] kv-cache update ownership moved to OV "
                "PagedAttention for %d attention layer(s)", _n)
        except Exception as _e:
            logger.debug(
                "[OV plugin] kv_cache_update suppression skipped: %s", _e)

        # Gated here rather than in register() so eager/inductor workers do not
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

        # Undo vLLM's 1-core pin before torch.compile so OV's pool inherits
        # the wide mask at creation.
        try:
            from openvino.frontend.pytorch.torchdynamo.vllm import compile_hooks as _vh_aff
            _vh_aff.widen_affinity_if_needed(None)
        except Exception as _e:
            logger.debug("[OV plugin] affinity widen skipped: %s", _e)

        logger.info("[OV plugin] Compiling model with torch.compile backend=openvino")
        # "vllm": True turns on every vLLM-required flag (see preset.py).
        # Precision keys are deliberately absent: derived per-model dtype.
        options = {"aot_autograd": True, "vllm": True}
        # dynamic=None: specializes first, then symbolizes only the dim that
        # varies -- cheaper than dynamic=True (~1.7x steady-state cost).
        compiled = torch.compile(
            self.model.forward,
            backend="openvino",
            fullgraph=False,
            dynamic=None,
            options=options,
        )
        self.model.forward = compiled

        # lm_head runs outside the compiled forward(); disabling onednn above
        # left it on plain F.linear, so give it back a fast GEMM (see lm_head.py).
        _install_ov_lm_head(self.model)

    CPUModelRunner.load_model = patched_load_model
    CPUModelRunner._ov_plugin_patched = True
    logger.debug("[OV plugin] CPUModelRunner.load_model patched")


def _patch_reload_weights():
    """Rebuild the OV lm_head after CPUModelRunner.reload_weights.

    Only lm_head needs this: forward() reads Parameters fresh each call, but
    build_ov_lm_head baked the old weight into an OV constant at load time.
    """
    try:
        from vllm.v1.worker.cpu_model_runner import CPUModelRunner
    except Exception:
        return

    if getattr(CPUModelRunner, "_ov_plugin_reload_patched", False):
        return

    _orig_reload_weights = CPUModelRunner.reload_weights

    def patched_reload_weights(self, *args, **kwargs):
        _orig_reload_weights(self, *args, **kwargs)
        if _ov_active(self):
            _install_ov_lm_head(self.model)

    CPUModelRunner.reload_weights = patched_reload_weights
    CPUModelRunner._ov_plugin_reload_patched = True
    logger.debug("[OV plugin] CPUModelRunner.reload_weights patched")


def _patch_worker_update_weights():
    """Rebuild the OV lm_head after CPUWorker.update_weights (RLHF weight sync).

    Lives on the worker, not the model runner, so it needs its own patch.
    """
    try:
        from vllm.v1.worker.cpu_worker import CPUWorker
    except Exception:
        return

    if getattr(CPUWorker, "_ov_plugin_update_patched", False):
        return

    _orig_update_weights = CPUWorker.update_weights

    def patched_update_weights(self, *args, **kwargs):
        _orig_update_weights(self, *args, **kwargs)
        model_runner = getattr(self, "model_runner", None)
        if model_runner is not None and _ov_active(model_runner):
            _install_ov_lm_head(model_runner.get_model())

    CPUWorker.update_weights = patched_update_weights
    CPUWorker._ov_plugin_update_patched = True
    logger.debug("[OV plugin] CPUWorker.update_weights patched")


def _check_layername():
    """Warn if vLLM latched VLLM_USE_LAYERNAME=1 before this plugin loaded.

    Too late to fix here -- vllm.utils.torch_utils freezes it at import,
    before any entry point loads -- so just warn instead of an obscure
    AttributeError later in openvino_compile.
    """
    try:
        from vllm.utils.torch_utils import _USE_LAYERNAME
    except Exception:
        return
    if not _USE_LAYERNAME:
        return
    logger.warning(
        "[OV plugin] VLLM_USE_LAYERNAME is enabled; the OV backend cannot "
        "compile graphs whose layer_name is a hoisted LayerName input and "
        "will fail with \"'LayerName' object has no attribute 'type'\". Call "
        "openvino.frontend.pytorch.torchdynamo.vllm.preset.set_pre_import_env() "
        "before importing vllm. Setting the env var now has no effect: vLLM "
        "reads it at import, before plugins load."
    )


def _apply_default_env():
    """Fill in unset env vars only; any user-set value takes priority."""
    defaults = {
        # KV-cache pool size (GiB). Zero means "fill node memory", which
        # OOM-kills on shared NUMA nodes. 4 covers 1-2B models at 2k ctx.
        "VLLM_CPU_KVCACHE_SPACE": "4",
        # execute.py fast path caching set_tensor bindings and output views
        # across steps. +5-15% greedy, no correctness regression.
        "OV_FAST_INFER": "1",
    }
    for key, value in defaults.items():
        if os.environ.get(key) is None:
            os.environ[key] = value
            logger.debug("[OV plugin] %s=%s (default)", key, value)


def _warn_if_unpinned():
    """Warn when the process is not pinned to a subset of CPUs.

    Full-machine affinity lets threads land on multiple NUMA nodes, which
    halves decode memory bandwidth. Best-effort: silently skipped where
    sched_getaffinity is unavailable (macOS, Windows).
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

    OV-active detection is deferred to patched_load_model, which can read the
    real compilation_config.backend instead of guessing from env vars.
    """
    _apply_default_env()
    _check_layername()
    _warn_if_unpinned()
    _patch_cpu_model_runner()
    _patch_reload_weights()
    _patch_worker_update_weights()
