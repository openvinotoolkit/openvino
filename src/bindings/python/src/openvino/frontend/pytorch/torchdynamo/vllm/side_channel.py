# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""vLLM PagedAttention side-channel binding.

The C++ paged_attention translator emits extra OV Parameters named
"__pa__<layer>__<field>" (key_cache, value_cache, past_lens, ...). At infer
time we resolve those from vllm.forward_context and bind them as side-channel
inputs.
"""

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Per-layer KV-cache ov.Tensor wrappers, keyed by meta layer name so they
# persist across torch-compile invocations. The underlying torch tensors live
# for the whole generate call, so wrap once and reuse.
_pa_kv_ovt_cache = {}
_pa_sliding_window_cache = {}  # (id(compiled), layer_name) -> np.int32 array
# (id(compiled), layer_name) -> {meta_layer_name, layer_obj, kv_cache_id}.
# Architecture-static within a generate() call, so caching it skips the
# per-step _placeholder_to_real / nc_layers.get / kv_sharing chase.
# kv_cache_id is the invalidation key: vLLM may re-allocate the cache.
_pa_layer_static_cache = {}


# Zero placeholders for when vLLM has no attn_meta yet (e.g. warm-up
# dummy_run), so the happy path allocates nothing per decode step.
def _zeros_1_i32():
    return np.zeros(1, dtype=np.int32)


def _zeros_2_i32():
    return np.zeros(2, dtype=np.int32)


def _zero_scalar_i32():
    return np.array(0, dtype=np.int32)


# id(compiled) -> {"layer_to_fields": {layer_name: {field: parameter_name}}}.
# Built on first bind so the regex walk over compiled.inputs does not rerun on
# every decode step.
_pa_layout_cache = {}


def _pa_auto_detect_kv_geom(ctx, meta_layer_name, placeholder_layer_name=None):
    """Return (num_kv_heads, head_size) for the given layer.

    Tried in order: direct lookup in ctx.no_compile_layers; ordinal lookup by
    the placeholder's index ("unknown_layer_N" -> N), which is what works
    during warmup before meta_layer_name is resolvable from attn_metadata;
    global model_config; then (1, 1).

    The per-layer lookups matter because head size can vary within a model:
    Gemma-4-E2B has 28 local-attention layers at head_size=256 plus 7 global
    ones at 512, and a single model-wide geom sizes the KV buffer wrong for
    most of them.
    """
    def _extract(layer_obj):
        try:
            hk = int(getattr(layer_obj, "num_kv_heads", 0)) or 0
            hs = int(getattr(layer_obj, "head_size", 0)) or 0
        except Exception:
            return None
        return (hk, hs) if hk and hs else None

    # 1. Direct, by resolved real layer name.
    try:
        nc = ctx.no_compile_layers if ctx is not None else None
        if isinstance(nc, dict) and meta_layer_name is not None:
            got = _extract(nc.get(meta_layer_name))
            if got:
                return got
    except Exception:
        pass

    # 2. Ordinal lookup via placeholder index.
    try:
        nc = ctx.no_compile_layers if ctx is not None else None
        if isinstance(nc, dict) and nc and placeholder_layer_name is not None:
            import re as _re_ord
            m = _re_ord.match(r"unknown_layer(?:_(\d+))?$", placeholder_layer_name)
            if m is not None:
                idx = int(m.group(1)) if m.group(1) else 0
                keys = list(nc.keys())
                if idx < len(keys):
                    got = _extract(nc.get(keys[idx]))
                    if got:
                        return got
    except Exception:
        pass

    # 3. Global model config (uniform-geom fallback).
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


_PA_FIELDS = (
    "key_cache", "value_cache", "past_lens", "subsequence_begins",
    "block_indices", "block_indices_begins", "max_context_len",
    # Raw vLLM-format inputs. past_lens/subsequence_begins/max_context_len are
    # derived from these in-graph, so the translator emits these instead.
    "seq_lens", "query_start_loc",
    # Per-layer, from layer_obj.impl.sliding_window.
    "sliding_window",
)


def _bind_paged_attention_side_channel(compiled):
    """Resolve every "__pa__<layer>__<field>" input of `compiled`.

    Resolved against the current forward context, returning
    {parameter_name: array-or-ov.Tensor}. Relies on vLLM's
    CPUAttentionMetadata layout.
    """
    try:
        from vllm.forward_context import get_forward_context
    except Exception:
        return {}

    try:
        ctx = get_forward_context()
    except AssertionError:
        # No ForwardContext (CPU warmup paths). Fall back to empty tensors so
        # PA at least does not segfault.
        ctx = None

    result = {}
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

    # A "shared" PA key (from get_or_make_shared_pa_param) can use any real
    # layer's attn_metadata: per-seq metadata is identical across layers.
    _first_real_layer = next(
        (ln for ln in layer_to_fields if ln != "shared"), None)

    # Real vLLM layer names, in the order the translator emitted its
    # placeholders (== model layer order).
    _real_layer_names = []
    if ctx is not None:
        try:
            _am = ctx.attn_metadata
            if isinstance(_am, dict):
                _real_layer_names = list(_am.keys())
                # attn_metadata groups layers by KV-cache spec (Gemma-4 hybrid:
                # all 28 sliding, then all 7 global), not by layer index, while
                # the frontend numbers PA nodes in FX-graph == model order. Sort
                # by the ".layers.<N>." index to match; fall back to dict order
                # for names without that pattern.
                import re as _re_sort

                def _layer_idx(name):
                    m = _re_sort.search(r"layers\.(\d+)", name)
                    return int(m.group(1)) if m else -1
                if all(_layer_idx(n) >= 0 for n in _real_layer_names):
                    _real_layer_names = sorted(_real_layer_names, key=_layer_idx)
        except Exception:
            pass

    def _placeholder_to_real(placeholder):
        """Map 'unknown_layer' -> real[0], 'unknown_layer_1' -> real[1], ...

        Modulo NUM_LAYERS: the translator's counter accumulates across
        torch-compile invocations (first compile 0..15, second 16..31, ...).
        """
        if not _real_layer_names:
            return None
        import re as _re_map
        m = _re_map.match(r"unknown_layer(?:_(\d+))?$", placeholder)
        if m is None:
            return None
        idx = int(m.group(1)) if m.group(1) else 0
        idx = idx % len(_real_layer_names)
        return _real_layer_names[idx]

    # Per-seq metadata is identical across layers within a forward pass:
    # compute once from any real layer's attn_meta, reuse for all bindings.
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
    _bi_cache = {}  # per-call block_indices dedup
    for layer_name, fields in layer_to_fields.items():
        attn_meta = None
        kv_cache = None
        _static_key = (id(compiled), layer_name)
        _static = _pa_layer_static_cache.get(_static_key)
        if _static is not None:
            # Invalidate if vLLM re-allocated this layer's kv_cache.
            _cached_layer_obj = _static["layer_obj"]
            if _cached_layer_obj is not None:
                try:
                    _cur_kv = _cached_layer_obj.kv_cache
                    if isinstance(_cur_kv, list):
                        _cur_kv = _cur_kv[ctx.virtual_engine] if ctx is not None else _cur_kv[0]
                    if id(_cur_kv) != _static["kv_cache_id"]:
                        _static = None
                except Exception:
                    _static = None

        if _static is None:
            # Slow path: resolve layer_obj, meta_layer_name, KV-sharing target.
            if layer_name == "shared":
                meta_layer_name = (
                    (_placeholder_to_real(_first_real_layer) if _first_real_layer else None)
                    or (_real_layer_names[0] if _real_layer_names else None)
                )
            else:
                meta_layer_name = _placeholder_to_real(layer_name) or layer_name
            layer_obj = None
            _resolved_kv_id = None
            if ctx is not None and layer_name != "shared":
                try:
                    nc_layers = ctx.no_compile_layers
                    layer_obj = nc_layers.get(meta_layer_name) if isinstance(nc_layers, dict) else None
                    # KV sharing (Gemma-4 hybrid): redirect to the target layer.
                    kv_sharing_tgt = (getattr(layer_obj, "kv_sharing_target_layer_name", None)
                                      if layer_obj is not None else None)
                    if kv_sharing_tgt is not None:
                        tgt_obj = nc_layers.get(kv_sharing_tgt) if isinstance(nc_layers, dict) else None
                        if tgt_obj is not None:
                            layer_obj = tgt_obj
                            meta_layer_name = kv_sharing_tgt
                    if layer_obj is not None and hasattr(layer_obj, "kv_cache"):
                        _kvc = layer_obj.kv_cache
                        if isinstance(_kvc, list):
                            _kvc = _kvc[ctx.virtual_engine]
                        _resolved_kv_id = id(_kvc)
                except Exception:
                    pass
            _static = {
                "meta_layer_name": meta_layer_name,
                "layer_obj": layer_obj,
                "kv_cache_id": _resolved_kv_id,
            }
            _pa_layer_static_cache[_static_key] = _static

        meta_layer_name = _static["meta_layer_name"]
        layer_obj = _static["layer_obj"]

        # Per-step attn_meta and kv_cache, via the cached refs.
        if ctx is not None:
            try:
                if meta_layer_name is not None:
                    am_map = ctx.attn_metadata
                    if isinstance(am_map, dict):
                        attn_meta = am_map.get(meta_layer_name)
                if layer_name != "shared" and layer_obj is not None and hasattr(layer_obj, "kv_cache"):
                    kv_cache = layer_obj.kv_cache
                    if isinstance(kv_cache, list):
                        kv_cache = kv_cache[ctx.virtual_engine]
            except Exception:
                pass

        key_cache_np = value_cache_np = None
        key_cache_ovt = value_cache_ovt = None
        if kv_cache is not None:
            try:
                # Key by real layer name so the OV tensor persists across
                # torch-compile invocations: prefill and decode share one
                # underlying buffer.
                cache_key = meta_layer_name
                cached = _pa_kv_ovt_cache.get(cache_key)
                if cached is not None:
                    key_cache_ovt, value_cache_ovt, kc, vc, key_cache_np, value_cache_np = cached
                else:
                    # vLLM 0.25's CPU KV cache is rank-4, [num_blocks,
                    # num_kv_heads, block_size, 2*head_size], K/V interleaved
                    # along the last dim: unbind(0) picks the wrong axis, so
                    # view + chunk is what splits it correctly.
                    if kv_cache.ndim == 4:
                        _nb, _hk, _bs, _last = kv_cache.shape
                        _view = kv_cache.view(_nb, _hk, _bs * 2, _last // 2)
                        kc, vc = _view.chunk(2, dim=2)
                        kc = kc.contiguous()
                        vc = vc.contiguous()
                    else:
                        kc, vc = kv_cache.unbind(0)
                    # OV-native Tensor matching the PA Parameter dtype; OV CPU
                    # PA writes back into this buffer via shared memory.
                    import openvino as _ov
                    _kv_shape = tuple(kc.shape)
                    # OV CPU PA hard-requires block_size==32. When vLLM uses
                    # B>32 (Gemma-4 hybrid unifies to 64 to keep page_size_bytes
                    # uniform across differing head_size), present the buffer as
                    # (N*ratio, Hk, 32, S) — a pure reshape, same element count.
                    # Block indices are re-expanded to match below.
                    if len(_kv_shape) >= 4 and _kv_shape[-2] > 32 and _kv_shape[-2] % 32 == 0:
                        _ratio = _kv_shape[-2] // 32
                        _param_shape = (_kv_shape[0] * _ratio,) + _kv_shape[1:-2] + (32, _kv_shape[-1])
                    else:
                        _param_shape = _kv_shape
                    # Take the dtype from this layer's own key_cache Parameter:
                    # the plugin may have retyped it via KV_CACHE_PRECISION.
                    _param_dt = None
                    _target_name = fields.get("key_cache", f"__pa__{layer_name}__key_cache")
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
            except Exception:
                pass
        if key_cache_np is None:
            # Fallback dummy. Hk and S must match what PA sees at real runtime,
            # or CPU PA caches Hk=1, S=1 from the dummy and then asserts against
            # the real K.
            import openvino as _ov_fb
            _fb_dt_ov = _ov_fb.Type.f32
            _fb_Hk, _fb_S = _pa_auto_detect_kv_geom(ctx, meta_layer_name, placeholder_layer_name=layer_name)
            _fb_block = 32  # CPU PA hard requirement
            _target_fb = fields.get("key_cache", f"__pa__{layer_name}__key_cache")
            for _pi in compiled.inputs:
                if _target_fb in _pi.get_names():
                    _fb_dt_ov = _pi.get_element_type()
                    break
            _fb_shape = (1, _fb_Hk, _fb_block, _fb_S)
            key_cache_ovt = _ov_fb.Tensor(_fb_dt_ov, _fb_shape)
            value_cache_ovt = _ov_fb.Tensor(_fb_dt_ov, _fb_shape)
            key_cache_np = key_cache_ovt.data if _fb_dt_ov != _ov_fb.Type.bf16 else None
            value_cache_np = value_cache_ovt.data if _fb_dt_ov != _ov_fb.Type.bf16 else None

        # Computed once, from the first layer that has attn_meta.
        if not _shared_built and attn_meta is not None:
            try:
                seq_lens = getattr(attn_meta, "seq_lens", None)
                qsl = getattr(attn_meta, "query_start_loc", None)
                if seq_lens is not None and qsl is not None:
                    _shared_meta["seq_lens_np"] = seq_lens.to(torch.int32).contiguous().numpy()
                    _shared_meta["qsl_np"] = qsl.to(torch.int32).contiguous().numpy()
                    q_lens = qsl[1:] - qsl[:-1]
                    _shared_meta["past_lens_np"] = (seq_lens - q_lens).to(torch.int32).contiguous().numpy()
                    _shared_meta["subseq_begins_np"] = _shared_meta["qsl_np"]
                    _shared_meta["max_ctx_len_np"] = np.array(int(seq_lens.max().item()), dtype=np.int32)
                _shared_built = True
            except Exception:
                pass

        # Per-layer, since models with multiple KV-cache groups (Gemma-4 hybrid)
        # have a distinct block_table per group. The (id(block_table),
        # block_size) cache makes this run once per distinct table per call:
        # once for uniform-attention models, twice for Gemma-4's two groups.
        block_indices_np = _zeros_1_i32()
        block_indices_begins_np = _zeros_2_i32()
        if attn_meta is not None and kv_cache is not None:
            try:
                seq_lens = getattr(attn_meta, "seq_lens", None)
                block_table = getattr(attn_meta, "block_table", None)
                if block_table is not None and seq_lens is not None:
                    block_size = int(kv_cache.shape[3]) if kv_cache.ndim >= 5 else 16
                    _bi_key = (id(block_table), block_size)
                    _bi_cached = _bi_cache.get(_bi_key)
                    if _bi_cached is not None:
                        block_indices_np, block_indices_begins_np = _bi_cached
                    else:
                        bt = block_table.to(torch.int32).contiguous()
                        blocks_per_seq = ((seq_lens + block_size - 1) // block_size).to(torch.int32)
                        rows = bt.shape[0] if bt.ndim > 0 else 1
                        bps_np = blocks_per_seq.numpy()
                        bt_np = bt.numpy()
                        _ov_ratio = block_size // 32 if block_size > 32 and block_size % 32 == 0 else 1
                        if rows > 0 and bps_np.sum() > 0:
                            max_blocks = bt_np.shape[1] if bt_np.ndim > 1 else bt_np.shape[0]
                            col_idx = np.arange(max_blocks, dtype=np.int32)
                            mask = col_idx[None, :] < bps_np[:, None]
                            flat_bt = bt_np.reshape(rows, -1) if bt_np.ndim > 1 else bt_np[None, :]
                            bi = flat_bt[mask].astype(np.int32, copy=False)
                            if _ov_ratio > 1:
                                _ov_offsets = np.arange(_ov_ratio, dtype=np.int32)[None, :]
                                bi = (bi[:, None] * _ov_ratio + _ov_offsets).reshape(-1)
                            block_indices_np = bi
                        begins = np.empty(rows + 1, dtype=np.int32)
                        begins[0] = 0
                        np.cumsum(bps_np * _ov_ratio, out=begins[1:])
                        block_indices_begins_np = begins
                        _bi_cache[_bi_key] = (block_indices_np, block_indices_begins_np)
            except Exception:
                pass

        # Architecture-static, so read from layer_obj.impl once and cache.
        _sw_key = (id(compiled), layer_name)
        sliding_window_np = _pa_sliding_window_cache.get(_sw_key)
        if sliding_window_np is None:
            sliding_window_val = 0
            if ctx is not None and layer_name != "shared":
                try:
                    nc_layers = ctx.no_compile_layers
                    lo = nc_layers.get(meta_layer_name) if isinstance(nc_layers, dict) else None
                    sw = None
                    if lo is not None:
                        impl = getattr(lo, "impl", None)
                        sw = getattr(impl, "sliding_window", None) if impl is not None else None
                        if sw is None:
                            sw = getattr(lo, "sliding_window", None)
                    if sw is not None:
                        # vLLM: a tuple (w-1, w-1) or (w-1, 0) on sliding
                        # layers, (-1, -1) on full attention. OV PA wants a
                        # scalar: N for "last N tokens", 0 for disabled.
                        if isinstance(sw, (tuple, list)):
                            sw = sw[0] if sw else -1
                        sw_int = int(sw)
                        sliding_window_val = 0 if sw_int < 0 else sw_int
                except Exception:
                    pass
            sliding_window_np = np.array(sliding_window_val, dtype=np.int32)
            _pa_sliding_window_cache[_sw_key] = sliding_window_np

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
                result[name] = block_indices_np
            elif field == "block_indices_begins":
                result[name] = block_indices_begins_np
            elif field == "max_context_len":
                result[name] = _sm["max_ctx_len_np"]
            elif field == "seq_lens":
                result[name] = _sm["seq_lens_np"]
            elif field == "query_start_loc":
                result[name] = _sm["qsl_np"]
            elif field == "sliding_window":
                result[name] = sliding_window_np

    return result
