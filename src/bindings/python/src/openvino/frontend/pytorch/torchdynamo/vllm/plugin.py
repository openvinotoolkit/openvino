# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""vLLM general plugin for the OpenVINO torch.compile backend.

Registered via the `vllm.general_plugins` entry point. Patches
`CPUModelRunner.load_model` to wire `torch.compile(backend="openvino")` when
the user asks for it, and forces `_supports_onednn=False` (onednn_mm
graph-breaks the OV trace and rejects f32 activations from AOT decomposition).
Also patches `CPUModelRunner.reload_weights` and `CPUWorker.update_weights` to
rebuild the OV lm_head after either fires -- it bakes the weight into an OV
constant, which would otherwise go stale on any post-load weight mutation.

This entry point loads after `import vllm`, so it cannot set env vars vLLM
latches at import time -- those live in preset.set_pre_import_env, which the
launching script must call itself.

No-op unless the user requested the OV backend.
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

    Called from patched_load_model and again from the reload_weights /
    update_weights patches below: build_ov_lm_head bakes the weight into an OV
    constant, so any later in-place weight mutation needs this re-run or the
    OV lm_head keeps serving logits off the stale copy with no error.

    Defaults to the oneDNN dispatch (OV_LM_HEAD=0). The OV lm_head is a second
    compiled model whose InferRequest is invoked between main-graph infers,
    and that interleaving roughly doubles the main graph's per-step time --
    measured on Llama-3.2-1B and Mistral-7B alike, and far larger than the
    OMP_NUM_THREADS sensitivity the OV path was introduced to avoid. The cost
    is not in the lm_head kernel itself (which is competitive) and does not
    respond to the second model's thread count or CPU pinning, so it is not
    tunable from here. Set OV_LM_HEAD=1 to opt back in.
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


def _patch_cpu_model_runner():
    try:
        from vllm.v1.worker.cpu_model_runner import CPUModelRunner
    except Exception:
        return

    if getattr(CPUModelRunner, "_ov_plugin_patched", False):
        return

    _orig_load_model = CPUModelRunner.load_model

    def patched_load_model(self, load_dummy_weights: bool = False) -> None:
        # Flip _supports_onednn BEFORE _orig_load_model so vLLM's FC layers see
        # the False value when they are constructed during model load.
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

        # Suppress vLLM's standalone unified_kv_cache_update op. OV's
        # PagedAttentionExtension performs the paged KV-cache write itself, so
        # vLLM's separate op is redundant here -- and having no OV translator,
        # it (plus the getitem nodes unpacking the K/V Results feeding it) is
        # left in eager PyTorch by the partitioner: 49 nodes per graph at
        # Llama-3.2-1B, which is the only thing keeping check_fully_supported()
        # from returning True. Suppressing it yields one fused OV partition
        # with zero eager leftovers.
        #
        # Attention.forward (attention.py:566) emits the op only when the
        # backend reports forward_includes_kv_cache_update=False.
        # CPUAttentionBackend sets False (cpu_attn.py:40) even though its impl's
        # forward() already calls ops.cpu_attn_reshape_and_cache under the same
        # guards; the AttentionBackend base default is True (backend.py:67).
        #
        # Must run BEFORE _orig_load_model: the flag is read at forward time
        # and _orig_load_model can trigger dummy/profile forwards.
        if is_ov:
            try:
                from vllm.v1.attention.backends import cpu_attn as _cpu_attn
                _cls = _cpu_attn.CPUAttentionBackend
                if not getattr(_cls, "_ov_plugin_kv_update_patched", False):
                    _cls.forward_includes_kv_cache_update = True
                    _cls._ov_plugin_kv_update_patched = True
                    logger.debug(
                        "[OV plugin] CPUAttentionBackend."
                        "forward_includes_kv_cache_update forced True "
                        "(OV PagedAttention owns the KV-cache write)")
            except Exception as _e:
                logger.debug(
                    "[OV plugin] kv_cache_update suppression skipped: %s", _e)

        _orig_load_model(self, load_dummy_weights)
        if not is_ov:
            return

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

        # Undo vLLM's init_cpu_threads_env pinning to a single core. Must run
        # before torch.compile so TBB/OV pools inherit the wide mask at
        # creation.
        try:
            from openvino.frontend.pytorch.torchdynamo.vllm import compile_hooks as _vh_aff
            _vh_aff.widen_affinity_if_needed(None)
        except Exception as _e:
            logger.debug("[OV plugin] affinity widen skipped: %s", _e)

        logger.info("[OV plugin] Compiling model with torch.compile backend=openvino")
        # "vllm": True turns on every vLLM-required flag (paged_attention,
        # pa_translate, unbind_affinity, no_fallback, fc_decompress) and seeds
        # DYNAMIC_QUANTIZATION_GROUP_SIZE=32. KV_CACHE_PRECISION and
        # INFERENCE_PRECISION_HINT are deliberately not seeded here: they are
        # derived per-model from the converted graph's own float dtype (see
        # preset.precision_config), always as a matching pair, because the OV
        # CPU PagedAttention kernel is only instantiated for matching
        # (compute, cache) types. Override any flag via `options`.
        options = {"aot_autograd": True, "vllm": True}
        # dynamic=None (torch's default), not False: under False every distinct
        # prefill token count is a guard failure costing a ~5.4 s retrace plus a
        # ~14 s OV compile_model, forever, under a varying request mix. None
        # specializes the first shape then lets automatic_dynamic_shapes make
        # the varying dim symbolic. Preferred over True, which marks every dim
        # rather than the ones that actually varied: ~5% over the static graph
        # in steady state vs True's ~1.7x. Measurements and the frontend fixes
        # the symbolic graph depends on: vllm/docs/dynamic_shapes.md.
        compiled = torch.compile(
            self.model.forward,
            backend="openvino",
            fullgraph=False,
            dynamic=None,
            options=options,
        )
        self.model.forward = compiled

        # lm_head runs OUTSIDE the compiled forward() (in compute_logits()), so
        # disabling onednn above left it on plain F.linear. Give it back a fast
        # GEMM: preferably OV's, which runs on OV's thread pool and so ignores
        # the narrow OMP_NUM_THREADS this path wants (see lm_head.py);
        # otherwise oneDNN's AMX path, faster per-thread but only when
        # OMP_NUM_THREADS happens to be tuned for this model and machine.
        _install_ov_lm_head(self.model)

    CPUModelRunner.load_model = patched_load_model
    CPUModelRunner._ov_plugin_patched = True
    logger.debug("[OV plugin] CPUModelRunner.load_model patched")


def _patch_reload_weights():
    """Rebuild the OV lm_head after CPUModelRunner.reload_weights.

    reload_weights overwrites model parameters in place (checkpoint reload,
    RLHF-style weight sync). The main forward() graph reads its weights fresh
    off the Parameter each call and needs no such hook; lm_head does, because
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

    update_weights lives on the worker, not the model runner -- it streams
    weight-transfer-engine chunks into the model -- so it needs its own patch
    rather than reusing _patch_reload_weights above.
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

    Setting the env var here would be useless: vllm.utils.torch_utils imports
    ahead of vllm.platforms and vllm.plugins, so by the time any entry-point
    group is loaded it has already frozen `_USE_LAYERNAME` and baked
    LayerNameType into the custom-op schemas. The only fix is to set the var
    before `import vllm` (see preset.set_pre_import_env), so say so here rather
    than let it surface as an AttributeError inside openvino_compile.
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
    for k, v in defaults.items():
        if os.environ.get(k) is None:
            os.environ[k] = v
            logger.debug("[OV plugin] %s=%s (default)", k, v)


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
