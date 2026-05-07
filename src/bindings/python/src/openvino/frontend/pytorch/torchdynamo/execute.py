# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# mypy: ignore-errors

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from warnings import warn

import torch
import torch.overrides

from torch.fx import GraphModule
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from openvino.frontend import FrontEndManager
from openvino.frontend.pytorch.fx_decoder import TorchFXPythonDecoder
from openvino.frontend.pytorch.torchdynamo.partition import Partitioner
from openvino.frontend.pytorch.torchdynamo.compile import openvino_compile
from openvino import Core, Type, PartialShape
from openvino.frontend.pytorch.torchdynamo.backend_utils import _get_cache_dir, _get_device, _get_aot_autograd

from typing import Optional, Any

from torch.fx.experimental.proxy_tensor import make_fx, wrapper_and_args_for_make_fx

import logging
logger = logging.getLogger(__name__)


DEFAULT_OPENVINO_PYTHON_CONFIG = MappingProxyType(
    {
        "use_python_fusion_cache": True,
        "allow_single_op_fusion": True,
    },
)

compiled_cache = {}
req_cache = {}
max_openvino_partitions = 0
partitioned_modules = {}
# Cache keyed by structural hash of submodule FX graph + input dtype/rank.
# This lets us reuse a compiled OV model across dynamo retraces where the
# graph is structurally identical (typical during decode loops).
structural_cache = {}
# Per-layer KV-cache ov.Tensor wrappers. The underlying torch tensors live for
# the whole generate call, so we wrap them once and reuse. Keyed by (layer_name,
# id(kv_cache_tensor)) so it invalidates if vLLM rebuilds the KV allocator.
_pa_kv_ovt_cache = {}


# Small helpers for the zero placeholders we return when vLLM has no attn_meta
# yet (e.g. warm-up dummy_run). These are pre-built at import time so we don't
# allocate on every decode step in the happy path.
def _zeros_1_i32():
    return np.zeros(1, dtype=np.int32)
def _zeros_2_i32():
    return np.zeros(2, dtype=np.int32)
def _zero_scalar_i32():
    return np.array(0, dtype=np.int32)

# Per-compiled-model layout cache for PA side-channel binding. Maps
# id(compiled) -> {
#   "layers":       list of (layer_name, meta_layer_name, {field: parameter_name}),
#   "first_real":   placeholder->real layer_name for "shared" fallback,
#   "real_names":   vLLM real layer name list used for placeholder->real mapping,
# }
# Built lazily on first bind and reused thereafter. Kills the regex + iterate
# over compiled.inputs cost on every decode step.
_pa_layout_cache = {}


def _pa_auto_detect_kv_geom(ctx, meta_layer_name):
    """Return (num_kv_heads, head_size) for the model.

    Order of preference:
      1. vLLM Attention layer object in ctx.no_compile_layers (authoritative).
      2. vLLM global model_config via get_current_vllm_config().
      3. Fallback (1, 1) — caller can still override via env.
    """
    # 1. Via Attention layer
    try:
        nc_layers = ctx.no_compile_layers if ctx is not None else None
        layer_obj = nc_layers.get(meta_layer_name) if isinstance(nc_layers, dict) else None
        if layer_obj is None and isinstance(nc_layers, dict) and nc_layers:
            for _v in nc_layers.values():
                if hasattr(_v, "num_kv_heads") and hasattr(_v, "head_size"):
                    layer_obj = _v
                    break
        if layer_obj is not None:
            hk = int(getattr(layer_obj, "num_kv_heads", 0)) or 0
            hs = int(getattr(layer_obj, "head_size", 0)) or 0
            if hk and hs:
                return hk, hs
    except Exception:
        pass
    # 2. Via global vLLM config
    try:
        from vllm.config import get_current_vllm_config
        cfg = get_current_vllm_config()
        mc = cfg.model_config
        pc = cfg.parallel_config
        hk = int(mc.get_num_kv_heads(pc))
        hs = int(mc.get_head_size())
        if hk and hs:
            return hk, hs
    except Exception:
        pass
    return 1, 1


def _structural_key(gm, args):
    """Structural hash of the FX graph that's stable across re-traces.

    Uses normalized node ops + consumer chain instead of gm.code (which has
    arbitrary name suffixes like 'arg99_1' vs 'arg132_1' that differ across
    traces despite identical structure).
    """
    try:
        parts = []
        # Assign index-based ids to placeholders so arg99_1/arg132_1 don't
        # produce different hashes for structurally identical graphs.
        node_id = {}
        ph_i = 0
        for n in gm.graph.nodes:
            if n.op == "placeholder":
                node_id[n] = f"ph{ph_i}"
                ph_i += 1
                parts.append(f"placeholder")
                continue
            # node target (stable)
            t = str(n.target) if hasattr(n, 'target') else str(n.op)
            # input edge descriptor: refer by node_id if known, else by op
            arg_descs = []
            for a in n.args:
                arg_descs.append(node_id.get(a, type(a).__name__))
            parts.append(f"{n.op}:{t}({','.join(arg_descs)})")
            node_id[n] = f"n{len(node_id)}"
    except Exception:
        parts = [str(id(gm))]
    sig = ["|".join(parts)]
    for a in args:
        if isinstance(a, torch.Tensor):
            sig.append(f"T{a.dtype}:{tuple(a.size())}")
        elif isinstance(a, int):
            sig.append(f"I:{a}")
        else:
            sig.append(f"S{type(a).__name__}")
    import hashlib
    return hashlib.sha256("|".join(sig).encode()).hexdigest()


def execute(
    gm: GraphModule,
    *args,
    executor: str = "openvino",
    executor_parameters: Optional[dict] = None,
    options: Optional[Any] = None,
):
    if executor == "openvino":
        return openvino_execute_partitioned(
            gm, *args, executor_parameters=executor_parameters, options=options
        )
    elif executor == "strictly_openvino":
        return openvino_execute(gm, *args, executor_parameters=executor_parameters)

    msg = (
        "Received unexpected value for 'executor': {0}. "
        "Allowed values are: openvino, strictly_openvino."
    ).format(executor)
    raise ValueError(msg)


import numpy as np


_PA_FIELDS = (
    "key_cache", "value_cache", "past_lens", "subsequence_begins",
    "block_indices", "block_indices_begins", "max_context_len",
    # Raw vLLM-format inputs: past_lens/subsequence_begins/max_context_len
    # are now *derived* from these in-graph via SDPAToPagedAttention-style
    # ops, so the translator emits these Parameters instead.
    "seq_lens", "query_start_loc",
)


def _bind_paged_attention_side_channel(compiled):
    """For every compiled-model input named "__pa__<layer>__<field>", look up
    the tensor from vllm.forward_context.get_forward_context() and return a
    mapping {name: numpy_array}.

    Relies on vLLM CPU attention metadata layout (CPUAttentionMetadata).
    """
    try:
        from vllm.forward_context import get_forward_context
    except Exception:
        return {}

    try:
        ctx = get_forward_context()
    except AssertionError:
        # No ForwardContext set (e.g. during CPU warmup paths); fall back to
        # empty tensors so PA at least doesn't segfault.
        ctx = None

    result = {}
    # Group the PA inputs by layer_name — cache across calls on compiled id.
    _layout = _pa_layout_cache.get(id(compiled))
    if _layout is None:
        layer_to_fields = {}
        import re as _re_pa
        _suffix_re = _re_pa.compile(r"_(\d+)$")
        for inp in compiled.inputs:
            for nm in inp.get_names():
                if not nm.startswith("__pa__"):
                    continue
                rest = nm[len("__pa__"):]
                _sm = _suffix_re.search(rest)
                rest_stripped = rest[:_sm.start()] if _sm else rest
                layer_suffix = _sm.group(0) if _sm else ""
                for field in _PA_FIELDS:
                    suffix = "__" + field
                    if rest_stripped.endswith(suffix):
                        layer_name = rest_stripped[: -len(suffix)] + layer_suffix
                        layer_to_fields.setdefault(layer_name, {})[field] = nm
                        break
                break
        _layout = {"layer_to_fields": layer_to_fields}
        _pa_layout_cache[id(compiled)] = _layout
    layer_to_fields = _layout["layer_to_fields"]

    # If a "shared" PA key exists (from get_or_make_shared_pa_param), we treat
    # any real layer's attn_metadata as representative since per-seq metadata
    # is identical across layers.
    _first_real_layer = next(
        (ln for ln in layer_to_fields if ln != "shared"), None)

    # Build mapping: our placeholder layer_names like "unknown_layer",
    # "unknown_layer_1", ... -> vLLM real layer names from attn_metadata, in
    # the order the translator emitted them (== model layer order).
    _real_layer_names = []
    if ctx is not None:
        try:
            _am = ctx.attn_metadata
            if isinstance(_am, dict):
                _real_layer_names = list(_am.keys())
        except Exception:
            pass

    def _placeholder_to_real(placeholder):
        """Map 'unknown_layer' -> real[0], 'unknown_layer_1' -> real[1], ...
        Modulo NUM_LAYERS because translator counter accumulates across
        torch-compile invocations (first compile 0..15, second 16..31, ...)."""
        if not _real_layer_names:
            return None
        import re as _re_map
        m = _re_map.match(r"unknown_layer(?:_(\d+))?$", placeholder)
        if m is None:
            return None
        idx = int(m.group(1)) if m.group(1) else 0
        idx = idx % len(_real_layer_names)
        return _real_layer_names[idx]

    # Per-seq metadata (seq_lens, qsl, block_indices, etc.) is identical across
    # all layers in a single forward pass. Compute it once from any real
    # layer's attn_meta, then reuse for every layer's field binding.
    _shared_meta = {
        "past_lens_np": _zeros_1_i32(),
        "subseq_begins_np": _zeros_2_i32(),
        "block_indices_np": _zeros_1_i32(),
        "block_indices_begins_np": _zeros_2_i32(),
        "max_ctx_len_np": _zero_scalar_i32(),
        "seq_lens_np": _zeros_1_i32(),
        "qsl_np": _zeros_2_i32(),
    }
    _shared_built = False
    for layer_name, fields in layer_to_fields.items():
        attn_meta = None
        kv_cache = None
        # For the shared Parameter group, fall back to any real layer's
        # attn_metadata (per-seq fields are identical across layers).
        if layer_name == "shared":
            # _first_real_layer is a placeholder like "unknown_layer"; map it to
            # a real vLLM layer name via the same mapping as per-layer params.
            meta_layer_name = (_placeholder_to_real(_first_real_layer) if _first_real_layer else None) \
                              or (_real_layer_names[0] if _real_layer_names else None)
        else:
            meta_layer_name = _placeholder_to_real(layer_name) or layer_name
        if ctx is not None:
            try:
                am_map = ctx.attn_metadata
                try:
                    import os as _osd
                    if _osd.environ.get("OV_DBG_PA"):
                        with open("/tmp/ov_path.log","a") as _fd:
                            _fd.write(f"am_map keys: {list(am_map.keys()) if isinstance(am_map, dict) else type(am_map)} layer_name={layer_name} meta={meta_layer_name}\n")
                except Exception:
                    pass
                if isinstance(am_map, dict) and meta_layer_name is not None:
                    attn_meta = am_map.get(meta_layer_name)
                # kv cache: vLLM stores it in static_forward_context (per layer).
                # For the shared group, KV isn't relevant — leave as None so it
                # falls through to dummy tensors (unused, layer-specific KV is
                # still bound by the per-layer key_cache/value_cache entries).
                if layer_name != "shared":
                    nc_layers = ctx.no_compile_layers
                    layer_obj = nc_layers.get(meta_layer_name) if isinstance(nc_layers, dict) else None
                    if layer_obj is not None and hasattr(layer_obj, "kv_cache"):
                        kv_cache = layer_obj.kv_cache
                        # kv_cache may be list indexed by virtual_engine
                        if isinstance(kv_cache, list):
                            kv_cache = kv_cache[ctx.virtual_engine]
            except Exception:
                pass

        # Prepare numpy arrays for each field
        def _nz(dtype, shape=(1,)):
            return np.zeros(shape, dtype=dtype)

        key_cache_np = value_cache_np = None
        key_cache_ovt = value_cache_ovt = None
        if kv_cache is not None:
            try:
                # Key by (layer_name, id(kv_cache)): per-compile placeholder is
                # fine within a single compile; cross-compile sharing is handled
                # below by keying on meta_layer_name too.
                # Key by meta_layer_name (vLLM's real layer name) so f32 buffer
                # persists across torch-compile invocations (prefill + decode).
                cache_key = meta_layer_name
                cached = _pa_kv_ovt_cache.get(cache_key)
                import os as _osdk
                if _osdk.environ.get("OV_DBG_CACHE"):
                    with open("/tmp/ov_cache.log","a") as _fk:
                        _fk.write(f"bind layer={layer_name} meta={meta_layer_name} HIT={cached is not None}\n")
                if cached is not None:
                    key_cache_ovt, value_cache_ovt, kc, vc, key_cache_np, value_cache_np = cached
                else:
                    kc, vc = kv_cache.unbind(0)
                    # Allocate OV-native f32 Tensor (matches PA Parameter dtype).
                    # OV CPU PA writes back to this buffer via shared_memory.
                    import openvino as _ov
                    _kv_shape = tuple(kc.shape)
                    import os as _os_kv
                    if _os_kv.environ.get("OV_DBG_KV_SHAPE"):
                        print(f"[KV_SHAPE] layer={layer_name} meta={meta_layer_name} "
                              f"kv_cache.shape={tuple(kv_cache.shape)} kc.shape={_kv_shape}", flush=True)
                    # Find the actual Parameter in compiled.inputs for this layer's
                    # key_cache and use its dtype (plugin may override via KV_CACHE_PRECISION).
                    _param_dt = None
                    _param_shape = _kv_shape
                    # Parameter name pattern: __pa__<layer_name>__key_cache
                    _target_name = f"__pa__{layer_name}__key_cache"
                    for _pi in compiled.inputs:
                        if _target_name in _pi.get_names():
                            _param_dt = _pi.get_element_type()
                            break
                    if _param_dt is None:
                        _param_dt = _ov.Type.f32
                    key_cache_ovt = _ov.Tensor(_param_dt, _param_shape)
                    value_cache_ovt = _ov.Tensor(_param_dt, _param_shape)
                    key_cache_np = key_cache_ovt.data
                    value_cache_np = value_cache_ovt.data
                    key_cache_np.fill(0)
                    value_cache_np.fill(0)
                    _pa_kv_ovt_cache[cache_key] = (
                        key_cache_ovt, value_cache_ovt, kc, vc, key_cache_np, value_cache_np)
                # KV buffers are kept alive in _pa_kv_ovt_cache; no need to
                # stash an extra list in the result dict.
            except Exception:
                pass
        if key_cache_np is None:
            # Fallback dummy — must provide (1, Hk, block_size, S) that match
            # what the PA op will see at real runtime. Otherwise CPU PA caches
            # Hk=1, S=1 from the dummy and asserts against the real K.
            # OV CPU PA requires block_size == 32 (hard constraint).
            import openvino as _ov_fb
            import os as _os_fb
            _fb_dt_ov = _ov_fb.Type.f32
            _fb_Hk, _fb_S = _pa_auto_detect_kv_geom(ctx, meta_layer_name)
            _fb_block = 32  # CPU PA hard requirement
            # Env overrides (for debugging / unusual models)
            if _os_fb.environ.get("OV_PA_NUM_KV_HEADS"):
                _fb_Hk = int(_os_fb.environ["OV_PA_NUM_KV_HEADS"])
            if _os_fb.environ.get("OV_PA_HEAD_SIZE"):
                _fb_S = int(_os_fb.environ["OV_PA_HEAD_SIZE"])
            if _os_fb.environ.get("OV_PA_BLOCK_SIZE"):
                _fb_block = int(_os_fb.environ["OV_PA_BLOCK_SIZE"])
            _target_fb = f"__pa__{layer_name}__key_cache"
            for _pi in compiled.inputs:
                if _target_fb in _pi.get_names():
                    _fb_dt_ov = _pi.get_element_type()
                    break
            _fb_shape = (1, _fb_Hk, _fb_block, _fb_S)
            if _os_fb.environ.get("OV_DBG_KV_SHAPE"):
                print(f"[KV_SHAPE] FALLBACK layer={layer_name} meta={meta_layer_name} "
                      f"shape={_fb_shape} dt={_fb_dt_ov}", flush=True)
            key_cache_ovt = _ov_fb.Tensor(_fb_dt_ov, _fb_shape)
            value_cache_ovt = _ov_fb.Tensor(_fb_dt_ov, _fb_shape)
            key_cache_np = key_cache_ovt.data if _fb_dt_ov != _ov_fb.Type.bf16 else None
            value_cache_np = value_cache_ovt.data if _fb_dt_ov != _ov_fb.Type.bf16 else None

        # Per-seq metadata: build once per forward pass from the first
        # layer that has real attn_meta; all other layers reuse it.
        if not _shared_built and attn_meta is not None:
            try:
                seq_lens = getattr(attn_meta, "seq_lens", None)
                qsl = getattr(attn_meta, "query_start_loc", None)
                block_table = getattr(attn_meta, "block_table", None)
                if seq_lens is not None and qsl is not None:
                    _shared_meta["seq_lens_np"] = seq_lens.to(torch.int32).contiguous().numpy()
                    _shared_meta["qsl_np"] = qsl.to(torch.int32).contiguous().numpy()
                    q_lens = qsl[1:] - qsl[:-1]
                    _shared_meta["past_lens_np"] = (seq_lens - q_lens).to(torch.int32).contiguous().numpy()
                    _shared_meta["subseq_begins_np"] = _shared_meta["qsl_np"]
                    _shared_meta["max_ctx_len_np"] = np.array(int(seq_lens.max().item()), dtype=np.int32)
                if block_table is not None and seq_lens is not None:
                    bt = block_table.to(torch.int32).contiguous()
                    block_size = int(kv_cache.shape[3]) if kv_cache is not None and kv_cache.ndim >= 5 else 16
                    blocks_per_seq = ((seq_lens + block_size - 1) // block_size).to(torch.int32)
                    rows = bt.shape[0] if bt.ndim > 0 else 1
                    # CSR trim: take bt[i, :blocks_per_seq[i]] for each i.
                    # Fastest vectorized form using numpy (bt is small).
                    bps_np = blocks_per_seq.numpy()
                    bt_np = bt.numpy()
                    if rows > 0 and bps_np.sum() > 0:
                        # Build a flat index mask without a Python loop.
                        max_blocks = bt_np.shape[1] if bt_np.ndim > 1 else bt_np.shape[0]
                        col_idx = np.arange(max_blocks, dtype=np.int32)
                        # mask[i, j] = 1 if j < bps_np[i]
                        mask = col_idx[None, :] < bps_np[:, None]  # [rows, max_blocks]
                        flat_bt = bt_np.reshape(rows, -1) if bt_np.ndim > 1 else bt_np[None, :]
                        _shared_meta["block_indices_np"] = flat_bt[mask].astype(np.int32, copy=False)
                    else:
                        _shared_meta["block_indices_np"] = np.zeros((0,), dtype=np.int32)
                    begins = np.empty(rows + 1, dtype=np.int32)
                    begins[0] = 0
                    np.cumsum(bps_np, out=begins[1:])
                    _shared_meta["block_indices_begins_np"] = begins
                _shared_built = True
            except Exception:
                pass

        import os as _osc
        if _osc.environ.get("OV_DBG_CACHE"):
            with open("/tmp/ov_cache.log", "a") as _fc:
                _kcshape = None if key_cache_np is None else key_cache_np.shape
                _kcmean = None if key_cache_np is None else float(np.abs(key_cache_np).mean())
                _fc.write(f"layer={layer_name} meta={meta_layer_name} kc_mean={_kcmean} kc_shape={_kcshape} kv_cache_found={kv_cache is not None} real_names={len(_real_layer_names)}\n")
        _sm = _shared_meta
        for field, name in fields.items():
            if field == "key_cache":
                result[name] = key_cache_ovt if key_cache_ovt is not None else key_cache_np
            elif field == "value_cache":
                result[name] = value_cache_ovt if value_cache_ovt is not None else value_cache_np
            elif field == "past_lens":
                result[name] = _sm["past_lens_np"]
            elif field == "subsequence_begins":
                result[name] = _sm["subseq_begins_np"]
            elif field == "block_indices":
                result[name] = _sm["block_indices_np"]
            elif field == "block_indices_begins":
                result[name] = _sm["block_indices_begins_np"]
            elif field == "max_context_len":
                result[name] = _sm["max_ctx_len_np"]
            elif field == "seq_lens":
                result[name] = _sm["seq_lens_np"]
            elif field == "query_start_loc":
                result[name] = _sm["qsl_np"]

    return result


def execute_cached(compiled_model, *args):
    import os as _os_ec
    ov_inputs = [a.detach().cpu().numpy() for a in args]
    ov_inputs.reverse()
    if _os_ec.environ.get("OV_PERF_COUNT"):
        _req = compiled_model.create_infer_request()
        _res = _req.infer(ov_inputs)
        _pc_path = _os_ec.environ.get("OV_PERF_COUNT_OUT", "/tmp/ov_perf_count.log")
        try:
            with open(_pc_path, "a") as _fpc:
                _fpc.write(f"# START execute_cached\n")
                for _p in _req.profiling_info:
                    _fpc.write(f"{_p.node_type}\t{_p.node_name}\t{_p.real_time.total_seconds()*1e6:.2f}\t{_p.cpu_time.total_seconds()*1e6:.2f}\t{_p.exec_type}\n")
        except Exception:
            pass
        result = [torch.from_numpy(_res[out]) for out in compiled_model.outputs]
        return result
    res = compiled_model(ov_inputs)
    result = [torch.from_numpy(res[out]) for out in compiled_model.outputs]
    return result


def openvino_execute(
    gm: GraphModule,
    *args,
    executor_parameters=None,
    partition_id: int = 0,
    options=None,
):

    executor_parameters = executor_parameters or DEFAULT_OPENVINO_PYTHON_CONFIG

    use_cache = executor_parameters.get(
        "use_python_fusion_cache",
        DEFAULT_OPENVINO_PYTHON_CONFIG["use_python_fusion_cache"],
    )
    global compiled_cache  # noqa: F824

    model_hash_str = executor_parameters.get("model_hash_str", None)
    if model_hash_str is not None:
        fully_supported = False
        if len(model_hash_str) > 3 and model_hash_str[-3:] == "_fs":
            fully_supported = True
        if not fully_supported:
            model_hash_str = model_hash_str + "_p" + str(partition_id)

    # Include input shape in the cache key: OV bakes concrete shapes into the
    # compiled model, so reusing a compiled partition with different input
    # shapes yields zero-sized outputs (observed on vLLM decode where each
    # step has a different seq-length).
    shape_key = tuple(
        tuple(a.size()) if isinstance(a, torch.Tensor) else (type(a).__name__, a)
        for a in args
    )
    cache_key = (partition_id, shape_key)

    if use_cache and (cache_key in compiled_cache):
        compiled = compiled_cache[cache_key]
        req = req_cache[cache_key]
    else:
        struct_key = _structural_key(gm, args)
        if use_cache and struct_key in structural_cache:
            compiled, req = structural_cache[struct_key]
        else:
            import time as _t_c0, os as _os_c0
            _tc_s = _t_c0.perf_counter()
            compiled = openvino_compile(gm, *args, model_hash_str=model_hash_str, options=options)
            req = compiled.create_infer_request()
            _tc_e = _t_c0.perf_counter()
            if _os_c0.environ.get("OV_TIME", "1") == "1":
                with open("/tmp/ov_timing.log", "a") as _ftc:
                    _shapes = [tuple(a.size()) if hasattr(a,'size') else a for a in args]
                    _ftc.write(f"COMPILE: {1000*(_tc_e-_tc_s):.1f}ms shapes={_shapes[:3]}\n")
            structural_cache[struct_key] = (compiled, req)
        compiled_cache[cache_key] = compiled
        req_cache[cache_key] = req

    flat_args, _ = tree_flatten(args)
    ov_inputs = []
    for arg in flat_args:
        if isinstance(arg, int):
            # Int args are baked into the compiled OV model as Constants
            # (see compile.py). Don't pass them at infer time.
            continue
        t = arg.detach()
        if not t.is_contiguous():
            t = t.contiguous()
        ov_inputs.append(t.numpy())

    # Bind vLLM PagedAttention side-channel Parameters (KV cache / block
    # tables / past_lens / ...) that the paged_attention C++ translator added
    # as extra Parameters with names like "__pa__<layer_name>__<field>".
    # The tensors come from vllm.forward_context.get_forward_context().
    import os as _os_ot, time as _t_ot
    _pr_ot = _os_ot.environ.get("OV_STEP_PROFILE")
    if _pr_ot: _t_bind_s = _t_ot.perf_counter()
    _pa_inputs_by_pos = {}
    if any(inp.get_names() and any(n.startswith("__pa__") for n in inp.get_names())
           for inp in compiled.inputs):
        _pa_inputs_by_pos = _bind_paged_attention_side_channel(compiled)
    if _pr_ot: _t_a = _t_ot.perf_counter()

    if _pa_inputs_by_pos:
        import os as _osd2
        if _osd2.environ.get("OV_DBG_PA"):
            with open("/tmp/ov_path.log", "a") as _f:
                _f.write(f"PA bind: compiled.inputs={len(compiled.inputs)} ov_inputs={len(ov_inputs)} pa_by_pos={len(_pa_inputs_by_pos)} flat_args={len(flat_args)}\n")
                for _ii, _inp in enumerate(compiled.inputs):
                    _nms = list(_inp.get_names())
                    _f.write(f"  [{_ii}] names={_nms} prec={_inp.get_element_type()} shape={_inp.get_partial_shape()}\n")
        # Use explicit set_tensor for PA inputs so the underlying shared-memory
        # buffer is bound directly; pass regular tensor inputs via dict.
        import openvino as _ov_bind
        _call_kwargs = {}
        _tensor_pos = 0
        import os as _os_dbg_vals
        _dbg_vals = _os_dbg_vals.environ.get("OV_DBG_BIND_VALUES")
        for i, inp in enumerate(compiled.inputs):
            _names = inp.get_names()
            pa_tensor = None
            for n in _names:
                if n.startswith("__pa__") and n in _pa_inputs_by_pos:
                    pa_tensor = _pa_inputs_by_pos[n]
                    break
            if pa_tensor is not None:
                _call_kwargs[inp] = pa_tensor
                if _dbg_vals:
                    with open("/tmp/ov_bind_vals.log", "a") as _fbv:
                        try:
                            if hasattr(pa_tensor, 'data') and not isinstance(pa_tensor, np.ndarray):
                                _arr = np.asarray(pa_tensor.data)
                            else:
                                _arr = np.asarray(pa_tensor)
                            _preview = _arr.flatten()[:8].tolist()
                            _fbv.write(f"BIND {n} shape={_arr.shape} dtype={_arr.dtype} first8={_preview}\n")
                        except Exception as _e:
                            _fbv.write(f"BIND {n} err: {_e}\n")
            else:
                _call_kwargs[inp] = ov_inputs[_tensor_pos]
                _tensor_pos += 1
        if _pr_ot: _t_b = _t_ot.perf_counter()
        _ti_s = _t_ot.perf_counter()
        res = req.infer(_call_kwargs, share_inputs=True, share_outputs=False)
        _ti_e = _t_ot.perf_counter()
        if _os_ot.environ.get("OV_TIME"):
            with open("/tmp/ov_timing.log", "a") as _fti:
                _fti.write(f"INFER: {1000*(_ti_e-_ti_s):.1f}ms\n")
        if _pr_ot:
            _t_c = _t_ot.perf_counter()
            with open("/tmp/ov_infer_prof.log", "a") as _fp_ot:
                _fp_ot.write(f"bind={1000*(_t_a-_t_bind_s):.2f}ms dict_build={1000*(_t_b-_t_a):.2f}ms infer={1000*(_t_c-_ti_s):.2f}ms\n")
    else:
        _ti2_s = _t_ot.perf_counter()
        res = req.infer(ov_inputs, share_inputs=True, share_outputs=False)
        _ti2_e = _t_ot.perf_counter()
        if _os_ot.environ.get("OV_TIME", "1") == "1":
            with open("/tmp/ov_timing.log", "a") as _fti2:
                _fti2.write(f"INFER: {1000*(_ti2_e-_ti2_s):.1f}ms\n")

    import os as _os_pc
    with open("/tmp/ov_path.log", "a") as _f: _f.write("openvino_execute infer done\n")
    # POST-infer cache logging removed: reading key_cache_ovt.data during
    # debug caused instability in the non-debug path.
    if _os_pc.environ.get("OV_PERF_COUNT"):
        _pc_path = _os_pc.environ.get("OV_PERF_COUNT_OUT", "/tmp/ov_perf_count.log")
        try:
            _pi = req.profiling_info
            with open(_pc_path, "a") as _fpc:
                _fpc.write(f"# START openvino_execute infer items={len(_pi)}\n")
                for _p in _pi:
                    _fpc.write(f"{_p.node_type}\t{_p.node_name}\t{_p.real_time.total_seconds()*1e6:.2f}\t{_p.cpu_time.total_seconds()*1e6:.2f}\t{_p.exec_type}\n")
        except Exception as _ee:
            with open(_pc_path, "a") as _fpc:
                _fpc.write(f"# ERROR: {_ee}\n")

    results1 = [torch.from_numpy(res[out]) for out in compiled.outputs]
    if len(results1) == 1:
        return results1[0]
    return results1


class OpenVINOGraphModule(torch.nn.Module):
    def __init__(self, gm, partition_id, use_python_fusion_cache, model_hash_str: str = None, options=None):
        super().__init__()
        self.gm = gm
        self.partition_id = partition_id
        self.executor_parameters = {"use_python_fusion_cache": use_python_fusion_cache,
                                    "model_hash_str": model_hash_str}
        self.perm_fallback = False
        self.options = options

    def __call__(self, *args):
        import os as _os_nof
        if self.perm_fallback and _os_nof.environ.get("OV_NO_FALLBACK", "0") != "1":
            return self.gm(*args)

        try:
            result = openvino_execute(
                self.gm,
                *args,
                executor_parameters=self.executor_parameters,
                partition_id=self.partition_id,
                options=self.options,
            )
            logger.debug("OpenVINO graph execution successful")
        except Exception as e:
            import os as _os
            import traceback as _tb
            with open("/tmp/ov_path.log", "a") as _f:
                _f.write(f"FALLBACK: {type(e).__name__}: {str(e)[:400]}\n")
                _f.write(_tb.format_exc() + "\n")
            if _os.environ.get("OV_TRACE_FALLBACK"):
                print(f"[OV_FALLBACK partition={self.partition_id}] {type(e).__name__}: {str(e)[:800]}", flush=True)
            if _os.environ.get("OV_NO_FALLBACK") == "1":
                raise  # Fail loudly so we can see where OV actually breaks
            logger.debug(
                f"OpenVINO execution failed with {e}. Falling back to native PyTorch execution."
            )
            self.perm_fallback = True
            return self.gm(*args)

        return result


def partition_graph(gm: GraphModule, use_python_fusion_cache: bool, model_hash_str: str = None, options=None):
    global max_openvino_partitions
    partition_id = max_openvino_partitions
    for node in gm.graph.nodes:
        # TODO: use a better way to identify fused submodule
        if node.op == "call_module" and "fused_" in node.name:
            openvino_submodule = getattr(gm, node.name)
            gm.delete_submodule(node.target)
            gm.add_submodule(
                node.target,
                OpenVINOGraphModule(openvino_submodule, partition_id, use_python_fusion_cache,
                                    model_hash_str=model_hash_str, options=options),
            )
            partition_id = partition_id + 1

    max_openvino_partitions = partition_id

    return gm


def openvino_execute_partitioned(gm: GraphModule, *args, executor_parameters=None, options=None):
    executor_parameters = executor_parameters or DEFAULT_OPENVINO_PYTHON_CONFIG

    global partitioned_modules  # noqa: F824

    use_python_fusion_cache = executor_parameters.get(
        "use_python_fusion_cache",
        DEFAULT_OPENVINO_PYTHON_CONFIG["use_python_fusion_cache"],
    )
    model_hash_str = executor_parameters.get("model_hash_str", None)

    signature = str(id(gm))
    if (not _get_aot_autograd(options)):
        for idx, input_data in enumerate(args):
            if isinstance(input_data, torch.Tensor):
                # Shape-agnostic: key only on dtype/rank so dynamic OV model is reused
                # across varying seq-lengths during decode instead of recompiling.
                signature = (
                    signature
                    + "_"
                    + str(idx)
                    + ":"
                    + str(input_data.type())[6:]
                    + ":rank"
                    + str(input_data.dim())
                )
            else:
                signature = (
                    signature
                    + "_"
                    + str(idx)
                    + ":"
                    + type(input_data).__name__
                )

    if signature not in partitioned_modules:
        partitioned_modules[signature] = partition_graph(
            gm, use_python_fusion_cache=use_python_fusion_cache, model_hash_str=model_hash_str, options=options
        )
    return partitioned_modules[signature](*args)


def clear_caches():
    global partitioned_modules  # noqa: F824
    global compiled_cache  # noqa: F824

    compiled_cache.clear()
    partitioned_modules.clear()
    _pa_kv_ovt_cache.clear()
