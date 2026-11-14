# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# mypy: ignore-errors

"""vLLM general plugin (`vllm.general_plugins`) wiring torch.compile(backend="openvino") into the CPU runner."""

import contextlib
import itertools
import logging
import os

logger = logging.getLogger(__name__)

# One id per compiled model for execute.py's caches; id()-based keys can be recycled after GC.
_next_model_id = itertools.count()


def _ov_active(self) -> bool:
    """True if `self.vllm_config.compilation_config` requests the OV backend."""
    return _ov_requested(getattr(self, "vllm_config", None))


def _ov_requested(vllm_config) -> bool:
    """True if `vllm_config.compilation_config` requests the OV backend."""
    comp_cfg = getattr(vllm_config, "compilation_config", None)
    if comp_cfg is None:
        return False
    try:
        mode = getattr(comp_cfg, "mode", None)
        mode_name = getattr(mode, "name", None) if mode is not None else None
        backend = getattr(comp_cfg, "backend", None)
    except Exception:
        return False
    return mode_name == "STOCK_TORCH_COMPILE" and backend == "openvino"


def _find_lm_head(model):
    """The model's lm_head; multimodal wrappers (e.g. Gemma 4) keep it on their language model."""
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None and hasattr(model, "get_language_model"):
        lm_head = getattr(model.get_language_model(), "lm_head", None)
    return lm_head


def _install_ov_lm_head(model) -> None:
    """(Re)build `lm_head.cpu_linear` from the current weight: oneDNN by default, OV with OV_LM_HEAD=1."""
    try:
        import vllm._custom_ops as _ops
        from vllm.model_executor.layers.utils import dispatch_cpu_unquantized_gemm
        from openvino.frontend.pytorch.torchdynamo.vllm.lm_head import build_ov_lm_head
        lm_head = _find_lm_head(model)
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
                    # Keep the weight: a later rebuild needs it.
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
    """`base` subclass with forward_includes_kv_cache_update=True, one per base (vLLM groups layers by class)."""
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


def _unsupported_attention_features(module) -> list:
    """Settings of one attention layer the OV PagedAttention translator would silently drop."""
    from vllm.v1.attention.backend import AttentionType
    impl = getattr(module, "impl", None)
    features = []
    if getattr(impl, "alibi_slopes", None) is not None:
        features.append("ALiBi slopes")
    if getattr(impl, "sinks", None) is not None:
        features.append("attention sinks")
    if getattr(impl, "logits_soft_cap", None):
        features.append("logits soft-capping")
    attn_type = getattr(impl, "attn_type", AttentionType.DECODER)
    if attn_type != AttentionType.DECODER:
        features.append(f"attention type {attn_type}")
    return features


def _own_kv_cache_update(model) -> int:
    """Let OV's PagedAttention own this model's KV-cache write, per layer (CPUAttentionBackend is process-global).

    Refuses a model with unsupported attention first: there is no falling back to vLLM's attention afterwards.
    """
    from vllm.v1.attention.backend import AttentionBackend
    layers = []
    for name, module in model.named_modules():
        # Some layers hold an AttentionBackendEnum here, not a backend class.
        backend = getattr(module, "attn_backend", None)
        if (isinstance(backend, type) and issubclass(backend, AttentionBackend)
                and not backend.forward_includes_kv_cache_update):
            layers.append((name, module, backend))
    unsupported = {name: features for name, module, _ in layers
                   if (features := _unsupported_attention_features(module))}
    if unsupported:
        first = next(iter(unsupported))
        raise NotImplementedError(
            f"The OpenVINO backend does not support {', '.join(unsupported[first])} in attention "
            f"(layer {first!r}, {len(unsupported)} layer(s) affected). Run this model without "
            f"the OpenVINO compilation backend.")
    for _, module, backend in layers:
        module.attn_backend = _kv_update_owned_backend(backend)
    return len(layers)


# Past the limit dynamo falls back to eager, which corrupts decode once OV owns the KV-cache write.
_DYNAMO_CACHE_SIZE_LIMIT_DEFAULT = 64


def _raise_dynamo_cache_limit():
    """Raise dynamo's cache_size_limit unless customized; OV_DYNAMO_FAIL_ON_CACHE_LIMIT=1 errors on exhaustion."""
    try:
        import torch._dynamo.config as _dynamo_cfg
    except Exception as _e:
        logger.debug("[OV plugin] dynamo cache_size_limit raise skipped: %s", _e)
        return
    limit = int(os.environ.get(
        "OV_DYNAMO_CACHE_SIZE_LIMIT", _DYNAMO_CACHE_SIZE_LIMIT_DEFAULT))
    if _dynamo_cfg.cache_size_limit == 8 and limit != 8:
        _dynamo_cfg.cache_size_limit = limit
        logger.debug("[OV plugin] dynamo cache_size_limit raised 8 -> %d", limit)
    if os.environ.get("OV_DYNAMO_FAIL_ON_CACHE_LIMIT", "0") != "0":
        _dynamo_cfg.fail_on_recompile_limit_hit = True
        logger.debug("[OV plugin] dynamo fail_on_recompile_limit_hit enabled")


def _compile_ov_forward(model, options) -> None:
    import torch
    model.forward = torch.compile(
        model._ov_eager_forward,
        backend="openvino",
        fullgraph=False,
        dynamic=None,
        options=options,
    )


def _rebuild_ov_model(runner, model) -> None:
    """Recompile the OV forward and lm_head after an in-place weight update, which frozen constants don't see."""
    recompiled = getattr(model, "_ov_eager_forward", None) is not None
    if recompiled:
        import torch
        from openvino.frontend.pytorch.torchdynamo import execute
        execute.clear_caches()
        # Dynamo's cached frames hold the old OV graphs; the sampler recompiles on first use.
        torch._dynamo.reset()
        options = dict(model._ov_compile_options, model_id=next(_next_model_id))
        _compile_ov_forward(model, options)
    _install_ov_lm_head(model)
    if recompiled and hasattr(runner, "warming_up_model"):
        # As at load, the warmup takes dynamo's static first trace, so OV compiles only the dynamic graph.
        runner.warming_up_model()


def _patch_cpu_model_runner():
    try:
        from vllm.v1.worker.cpu_model_runner import CPUModelRunner
    except Exception:
        return

    if getattr(CPUModelRunner, "_ov_plugin_patched", False):
        return

    _orig_load_model = CPUModelRunner.load_model

    def patched_load_model(self, load_dummy_weights: bool = False) -> None:
        # vLLM's FC layers read _supports_onednn at load; restore it for a later non-OV model.
        is_ov = _ov_active(self)
        if not is_ov:
            try:
                import vllm._custom_ops as _ops
                if getattr(_ops, "_ov_plugin_onednn_disabled", False):
                    _ops._supports_onednn = getattr(_ops, "_ov_plugin_onednn_saved", True)
                    _ops._ov_plugin_onednn_disabled = False
                    logger.debug("[OV plugin] _supports_onednn restored (backend != openvino)")
            except Exception as _e:
                logger.debug("[OV plugin] _supports_onednn restore skipped: %s", _e)
            _orig_load_model(self, load_dummy_weights)
            return

        try:
            import vllm._custom_ops as _ops
            if not getattr(_ops, "_ov_plugin_onednn_disabled", False):
                _ops._ov_plugin_onednn_saved = getattr(_ops, "_supports_onednn", True)
                _ops._supports_onednn = False
                _ops._ov_plugin_onednn_disabled = True
                logger.debug("[OV plugin] _supports_onednn forced False (backend=openvino)")
        except Exception as _e:
            logger.debug("[OV plugin] _supports_onednn flip skipped: %s", _e)

        _check_layername()
        _warn_if_unpinned()
        # vLLM picks each Linear's CPU kernel while loading weights.
        with _sgl_kernel_disabled():
            _orig_load_model(self, load_dummy_weights)

        options = {"aot_autograd": True, "vllm": True, "model_id": next(_next_model_id)}
        # The dtype vLLM built the model with, for the frontend decoder.
        _model_dtype = getattr(getattr(self, "model_config", None), "dtype", None)
        if _model_dtype is not None:
            options["default_float_type"] = _model_dtype

        from openvino.frontend.pytorch.torchdynamo.vllm import owns_kv_cache_update
        if owns_kv_cache_update(options):
            try:
                _n = _own_kv_cache_update(self.model)
                logger.debug(
                    "[OV plugin] kv-cache update ownership moved to OV "
                    "PagedAttention for %d attention layer(s)", _n)
            except NotImplementedError:
                raise  # attention the OV rewrite can't express: refuse the model
            except Exception as _e:
                logger.debug(
                    "[OV plugin] kv_cache_update suppression skipped: %s", _e)

        # Here, not in register(), so eager/inductor workers keep vLLM's sampler.
        try:
            from openvino.frontend.pytorch.torchdynamo.vllm import sampler as _vs
            _vs.install()
        except Exception as _e:
            logger.debug("[OV plugin] sampler install skipped: %s", _e)

        try:
            import openvino.torch  # noqa: F401  (registers backend)
        except Exception as e:
            logger.warning("OV plugin: failed to import openvino.torch: %s", e)
            return

        # Undo vLLM's 1-core pin before OV's thread pool is created.
        try:
            from openvino.frontend.pytorch.torchdynamo.vllm import compile_hooks as _vh_aff
            _vh_aff.widen_affinity_if_needed()
        except Exception as _e:
            logger.debug("[OV plugin] affinity widen skipped: %s", _e)

        _raise_dynamo_cache_limit()

        logger.info("[OV plugin] Compiling model with torch.compile backend=openvino")
        # Stops dynamo specializing num_tokens==1, so decode reuses prefill's compiled model.
        import torch.fx.experimental._config as _dynamo_cfg
        _dynamo_cfg.backed_size_oblivious = True
        self.model._ov_eager_forward = self.model.forward
        self.model._ov_compile_options = options
        _compile_ov_forward(self.model, options)

        # lm_head runs outside forward(), and disabling onednn left it on plain F.linear.
        _install_ov_lm_head(self.model)

    CPUModelRunner.load_model = patched_load_model
    CPUModelRunner._ov_plugin_patched = True
    logger.debug("[OV plugin] CPUModelRunner.load_model patched")


def _patch_reload_weights():
    """Rebuild the OV model and lm_head after CPUModelRunner.reload_weights."""
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
            _rebuild_ov_model(self, self.model)

    CPUModelRunner.reload_weights = patched_reload_weights
    CPUModelRunner._ov_plugin_reload_patched = True
    logger.debug("[OV plugin] CPUModelRunner.reload_weights patched")


def _patch_worker_update_weights():
    """Rebuild the OV model and lm_head at CPUWorker.finish_weight_update, once per RLHF sync, not per chunk."""
    try:
        from vllm.v1.worker.cpu_worker import CPUWorker
    except Exception:
        return

    if getattr(CPUWorker, "_ov_plugin_update_patched", False):
        return

    _orig_finish_weight_update = CPUWorker.finish_weight_update

    def patched_finish_weight_update(self, *args, **kwargs):
        _orig_finish_weight_update(self, *args, **kwargs)
        model_runner = getattr(self, "model_runner", None)
        if model_runner is not None and _ov_active(model_runner):
            _rebuild_ov_model(model_runner, model_runner.get_model())

    CPUWorker.finish_weight_update = patched_finish_weight_update
    CPUWorker._ov_plugin_update_patched = True
    logger.debug("[OV plugin] CPUWorker.finish_weight_update patched")


def _check_layername():
    """Warn if vLLM latched VLLM_USE_LAYERNAME=1 at import; too late to fix here."""
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
        # Caches input bindings and output views across steps; read only by the OV backend.
        "OV_FAST_INFER": "1",
    }
    for key, value in defaults.items():
        if os.environ.get(key) is None:
            os.environ[key] = value
            logger.debug("[OV plugin] %s=%s (default)", key, value)


def _warn_if_unpinned():
    """Warn when the process is not pinned: threads on several NUMA nodes cut decode memory bandwidth."""
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


# Only the KV pool; small bookkeeping tensors may rely on zeros.
_LAZY_KV_MIN_BYTES = 64 << 20


@contextlib.contextmanager
def _sgl_kernel_disabled():
    """Keep vLLM's CPU linears off the SGL kernel for one model load (OV_DISABLE_SGL_KERNEL=0 opts out).

    Its op is unsupported by the partitioner and would split the graph into one partition per layer.
    """
    if os.environ.get("OV_DISABLE_SGL_KERNEL", "1") == "0":
        logger.info("[OV plugin] SGL-kernel disable OFF (OV_DISABLE_SGL_KERNEL=0)")
        yield
        return
    try:
        from vllm.model_executor.layers import utils as _vllm_utils
        saved = _vllm_utils.check_cpu_sgl_kernel
    except Exception as _e:
        logger.debug("[OV plugin] disable-SGL patch skipped: %s", _e)
        yield
        return
    _vllm_utils.check_cpu_sgl_kernel = lambda *a, **kw: False
    logger.debug("[OV plugin] check_cpu_sgl_kernel forced to False for this model load")
    try:
        yield
    finally:
        _vllm_utils.check_cpu_sgl_kernel = saved


# KV-cache GiB for OV runs with no size set; vLLM's default fills node memory and OOMs on shared nodes.
_OV_DEFAULT_KVCACHE_GIB = 4


def _patch_kv_cache_default():
    """Default kv_cache_memory_bytes for OV runs, where vLLM applies VLLM_CPU_KVCACHE_SPACE."""
    try:
        from vllm.platforms.cpu import CpuPlatform
    except Exception as _e:
        logger.debug("[OV plugin] KV-cache default skipped: %s", _e)
        return
    if getattr(CpuPlatform, "_ov_plugin_kv_default_patched", False):
        return
    _orig = CpuPlatform.check_and_update_config.__func__

    def _patched(cls, vllm_config):
        _orig(cls, vllm_config)
        cache_config = getattr(vllm_config, "cache_config", None)
        if (cache_config is None or cache_config.kv_cache_memory_bytes is not None
                or not _ov_requested(vllm_config)):
            return
        cache_config.kv_cache_memory_bytes = _OV_DEFAULT_KVCACHE_GIB << 30
        logger.info(
            "[OV plugin] KV cache size defaults to %d GiB for the OpenVINO backend; "
            "set VLLM_CPU_KVCACHE_SPACE or kv_cache_memory_bytes to change it",
            _OV_DEFAULT_KVCACHE_GIB)

    CpuPlatform.check_and_update_config = classmethod(_patched)
    CpuPlatform._ov_plugin_kv_default_patched = True


def _patch_omp_wait_policy():
    """Spawn OV workers with OMP_WAIT_POLICY=PASSIVE unless the user set it.

    Spinning libgomp threads steal the cores OV's TBB workers are pinned to. Set at spawn, since libgomp reads it at
    load, and scoped like vLLM's own OMP variables, so other engines keep the spin.
    """
    try:
        from vllm.utils.ompmultiprocessing import OMPProcessManager
    except Exception as _e:
        logger.debug("[OV plugin] OMP wait-policy patch skipped: %s", _e)
        return
    if getattr(OMPProcessManager, "_ov_plugin_omp_patched", False):
        return
    _orig_init = OMPProcessManager.__init__
    _orig_configure = OMPProcessManager.configure_omp_envs

    def _patched_init(self, config):
        _orig_init(self, config)
        self._ov_requested = _ov_requested(config)

    @contextlib.contextmanager
    def _patched_configure(self, rank, local_rank):
        with _orig_configure(self, rank, local_rank):
            if not getattr(self, "_ov_requested", False) or "OMP_WAIT_POLICY" in os.environ:
                yield
                return
            os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
            try:
                yield
            finally:
                os.environ.pop("OMP_WAIT_POLICY", None)

    OMPProcessManager.__init__ = _patched_init
    OMPProcessManager.configure_omp_envs = _patched_configure
    OMPProcessManager._ov_plugin_omp_patched = True


def _patch_lazy_kv_cache():
    """Allocate the KV pool with torch.empty (OV_LAZY_KVCACHE=1): PA writes each block before reading it."""
    if os.environ.get("OV_LAZY_KVCACHE", "0") == "0":
        return
    try:
        import torch
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as _e:
        logger.debug("[OV plugin] lazy KV patch skipped: %s", _e)
        return

    if getattr(GPUModelRunner, "_ov_plugin_lazy_kv_patched", False):
        return

    _orig_alloc = GPUModelRunner._allocate_kv_cache_tensors

    def _patched_alloc(self, *args, **kwargs):
        if not _ov_active(self):
            return _orig_alloc(self, *args, **kwargs)
        # Only for the duration of this call.
        real_zeros = torch.zeros

        def _maybe_lazy_zeros(*a, **kw):
            size = a[0] if a else kw.get("size")
            nbytes = size if isinstance(size, int) else None
            if nbytes is not None and nbytes >= _LAZY_KV_MIN_BYTES:
                return torch.empty(*a, **kw)
            return real_zeros(*a, **kw)

        torch.zeros = _maybe_lazy_zeros
        try:
            return _orig_alloc(self, *args, **kwargs)
        finally:
            torch.zeros = real_zeros

    GPUModelRunner._allocate_kv_cache_tensors = _patched_alloc
    GPUModelRunner._ov_plugin_lazy_kv_patched = True
    logger.info("[OV plugin] KV cache pool allocated lazily (OV_LAZY_KVCACHE=1)")


def _register_quantization():
    """Register the NNCF quantization method and loader; unconditional, as vLLM reads them before load_model."""
    try:
        from openvino.frontend.pytorch.torchdynamo.vllm import quantization as _q
        _q.register()
    except Exception as _e:
        logger.debug("[OV plugin] quantization registration skipped: %s", _e)
    try:
        from openvino.frontend.pytorch.torchdynamo.vllm import model_loader as _ml
        _ml.register()
    except Exception as _e:
        logger.debug("[OV plugin] model loader registration skipped: %s", _e)


def register():
    """Entry point for `vllm.general_plugins`; runs in every process, so nothing here may affect non-OV runs."""
    _apply_default_env()
    _register_quantization()
    _patch_kv_cache_default()
    _patch_omp_wait_policy()
    _patch_cpu_model_runner()
    _patch_reload_weights()
    _patch_worker_update_weights()
    _patch_lazy_kv_cache()
