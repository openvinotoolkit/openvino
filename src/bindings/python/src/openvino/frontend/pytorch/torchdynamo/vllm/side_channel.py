# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""vLLM PagedAttention side-channel binding.

The C++ paged_attention frontend op emits extra OV Parameters with names
like \"__pa__<layer>__<field>\" (key_cache, value_cache, past_lens, ...).
At infer time we look those up in vllm.forward_context and bind them as
side-channel inputs. Lives in the vllm/ subpackage so torchdynamo.execute
does not need to import from vllm at all on standalone torch.compile.
"""

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

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


def _pa_auto_detect_kv_geom(ctx, meta_layer_name, placeholder_layer_name=None):
    """Return (num_kv_heads, head_size) for the given layer.

    Order of preference:
      1. Direct lookup by meta_layer_name in ctx.no_compile_layers.
      2. Ordinal lookup: derive the layer index from placeholder_layer_name
         (\"unknown_layer\" -> 0, \"unknown_layer_N\" -> N) and index into
         list(no_compile_layers.keys()). This is what works during warmup
         when meta_layer_name has not been resolved yet from attn_metadata.
      3. Global vLLM model_config (only correct when all layers share geom).
      4. Fallback (1, 1).

    Models like Gemma-4-E2B have mixed head sizes across layers (28 layers
    with head_size=256 for local attention + 7 with head_size=512 for
    global attention). Per-layer lookup via (2) is required; a single
    model-wide geom would size the KV buffer wrong for half the layers.
    """
    def _extract(layer_obj):
        try:
            hk = int(getattr(layer_obj, "num_kv_heads", 0)) or 0
            hs = int(getattr(layer_obj, "head_size", 0)) or 0
        except Exception:
            return None
        return (hk, hs) if hk and hs else None

    # 1. Direct lookup by resolved real layer name.
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
                if cached is not None:
                    key_cache_ovt, value_cache_ovt, kc, vc, key_cache_np, value_cache_np = cached
                else:
                    kc, vc = kv_cache.unbind(0)
                    # Allocate OV-native f32 Tensor (matches PA Parameter dtype).
                    # OV CPU PA writes back to this buffer via shared_memory.
                    import openvino as _ov
                    _kv_shape = tuple(kc.shape)
                    # Find the actual Parameter in compiled.inputs for this layer's
                    # key_cache and use its dtype (plugin may override via KV_CACHE_PRECISION).
                    _param_dt = None
                    _param_shape = _kv_shape
                    # Parameter name pattern: __pa__<layer_name>__key_cache
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
            _fb_dt_ov = _ov_fb.Type.f32
            _fb_Hk, _fb_S = _pa_auto_detect_kv_geom(ctx, meta_layer_name, placeholder_layer_name=layer_name)
            _fb_block = 32  # CPU PA hard requirement
            # Env overrides (for debugging / unusual models)
            import os as _os_fb
            if _os_fb.environ.get("OV_PA_NUM_KV_HEADS"):
                _fb_Hk = int(_os_fb.environ["OV_PA_NUM_KV_HEADS"])
            if _os_fb.environ.get("OV_PA_HEAD_SIZE"):
                _fb_S = int(_os_fb.environ["OV_PA_HEAD_SIZE"])
            if _os_fb.environ.get("OV_PA_BLOCK_SIZE"):
                _fb_block = int(_os_fb.environ["OV_PA_BLOCK_SIZE"])
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
