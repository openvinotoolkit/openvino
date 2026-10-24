# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Replaces vLLM's opaque _C.cpu_fused_moe with standard aten ops that OV lowers to GatherMatmul.

The ops form the tiled MoE block (every expert on every token, then a routing-weighted sum) that the CPU plugin's
ConvertTiledMoeBlockTo3GatherMatmuls matches. That pass turns it into three GatherMatmuls indexed by the [T, top_k]
router ids, so each matmul sees T rows. If the block doesn't match, it runs as dense batched matmuls over all experts.
"""

# mypy: ignore-errors

import logging
import weakref

from .fx_utils import is_auto_functionalized, packing_permutation, replace_mutated_output

logger = logging.getLogger(__name__)

_ACTIVATIONS = ("silu", "gelu", "gelu_tanh")
_perm_cache = {}
# Unpacked weights per packed storage, shared by every compiled graph of the model.
_unpacked_cache = weakref.WeakValueDictionary()
# Gate and up halves of an unpacked w13, keyed like _unpacked_cache plus the half.
_half_cache = weakref.WeakValueDictionary()
# Traced expert graphs per layer signature; every MoE layer of a model shares one.
_trace_cache = {}


def clear_weight_caches():
    """Drop the unpacked expert copies and halves: they are keyed by the packed address, which in-place reloads keep."""
    _unpacked_cache.clear()
    _half_cache.clear()


def _unpack_permutation(rows, cols, isa):
    """Source index per packed position of a [rows, cols] expert weight; None if the pack isn't a permutation."""
    import torch
    import vllm._custom_ops as ops
    key = (rows, cols, isa)
    if key not in _perm_cache:
        _perm_cache[key] = packing_permutation(lambda t: ops.cpu_prepack_moe_weight(t, isa),
                                               (1, rows, cols), torch.bfloat16)
    return _perm_cache[key]


def unpack_expert_weight(packed, isa):
    """Undo vLLM's cpu_prepack_moe_weight on a [experts, rows, cols] tensor."""
    import torch
    key = (packed.data_ptr(), tuple(packed.shape), isa)
    cached = _unpacked_cache.get(key)
    if cached is not None:
        return cached
    experts, rows, cols = packed.shape
    source = _unpack_permutation(rows, cols, isa)
    if source is None:
        return None
    plain = torch.empty_like(packed).reshape(experts, -1)
    plain[:, source] = packed.reshape(experts, -1)
    plain = plain.reshape(experts, rows, cols)
    _unpacked_cache[key] = plain
    return plain


def split_gate_up(packed, isa):
    """Unpacked, contiguous (gate, up) halves of a packed [experts, 2 * inter, hidden] w13; None if not unpackable."""
    key = (packed.data_ptr(), tuple(packed.shape), isa)
    halves = [_half_cache.get(key + (i,)) for i in (0, 1)]
    if any(h is None for h in halves):
        plain = unpack_expert_weight(packed, isa)
        if plain is None:
            return None
        half = plain.shape[1] // 2
        halves = [plain[:, :half].contiguous(), plain[:, half:].contiguous()]
        for i, weight in enumerate(halves):
            _half_cache[key + (i,)] = weight
    return halves


def moe_tiled(tokens, w1, w3, w2, topk_weights, topk_ids, act):
    """Compute MoE experts as a tiled block; same math as vLLM's cpu_fused_moe (w1 = gate, w3 = up).

    Shaped for ConvertTiledMoeBlockTo3GatherMatmuls: Tile -> Reshape -> gate/up MatMuls -> act * up -> down MatMul ->
    Reshape, times ScatterElementsUpdate -> Transpose -> Reshape routing weights, ReduceSum over experts.
    bmm and view on contiguous tensors keep -1 for the token dim in the trace (matmul and reshape would bake it in).
    """
    import torch
    from torch.nn import functional
    experts, hidden = w1.shape[0], tokens.shape[1]
    tiled = tokens.repeat(experts, 1).view(experts, -1, hidden)
    gate = torch.bmm(tiled, w1.transpose(1, 2))
    up = torch.bmm(tiled, w3.transpose(1, 2))
    if act == "silu":
        gated = functional.silu(gate) * up
    else:
        gated = functional.gelu(gate, approximate="tanh" if act == "gelu_tanh" else "none") * up
    down = torch.bmm(gated, w2.transpose(1, 2)).view(experts, -1, hidden)
    routing = torch.zeros_like(tokens[:, :1]).expand(-1, experts)
    routing = routing.scatter(1, topk_ids, topk_weights.to(tokens.dtype))
    routing = routing.transpose(0, 1).contiguous().view(experts, -1, 1)
    return (down * routing).sum(0)


def _constant(gm, node):
    """The tensor behind a get_attr node, or None for anything else."""
    if node.op != "get_attr":
        return None
    value = gm
    for part in node.target.split("."):
        value = getattr(value, part)
    return value


def _fused_moe_op():
    import torch
    return torch.ops._C.cpu_fused_moe.default


def _trace_expert_graph(act, x_dtype, weights, w_dtype, top_ids_val):
    """FX graph of moe_tiled for one layer signature, traced once and reused."""
    import torch
    from torch._guards import tracing
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch.utils._python_dispatch import _disable_current_modes

    w1, w3, w2 = weights
    top_k, id_dtype = top_ids_val.shape[1], top_ids_val.dtype
    key = (act, x_dtype, tuple(w1.shape), tuple(w2.shape), w1.dtype, w_dtype, top_k, id_dtype)
    if key not in _trace_cache:
        def body(tokens, w1, w3, w2, top_w, top_i):
            return moe_tiled(tokens, w1, w3, w2, top_w, top_i, act)

        # Fakes from a private static-shape mode, so dynamo's ShapeEnv gets no guards.
        with tracing(None), _disable_current_modes():
            fake_mode = FakeTensorMode(static_shapes=True)
            examples = [fake_mode.from_tensor(t, static_shapes=True) for t in (
                torch.empty(4, w1.shape[2], dtype=x_dtype), w1, w3, w2,
                torch.empty(4, top_k, dtype=w_dtype), torch.empty(4, top_k, dtype=id_dtype))]
            _trace_cache[key] = make_fx(body, tracing_mode="fake")(*examples)
    return _trace_cache[key]


def rewrite_cpu_fused_moe(gm) -> int:
    """Replace _C.cpu_fused_moe call sites with the tiled MoE formulation."""
    import torch
    from torch.utils._python_dispatch import _disable_current_modes

    nodes = [n for n in gm.graph.nodes if is_auto_functionalized(n, _fused_moe_op)]
    rewrites = 0
    for index, node in enumerate(nodes):
        kw = dict(node.kwargs)
        act = kw.get("act")
        top_ids = kw.get("topk_ids", kw.get("topk_id"))
        w13_node, w2_node = kw.get("w13"), kw.get("w2")
        if (act not in _ACTIVATIONS or top_ids is None or kw.get("skip_weighted")
                or kw.get("w13_bias") is not None or kw.get("w2_bias") is not None):
            logger.warning("fused_moe: leaving a call site on vLLM's op (act=%r)", act)
            continue
        packed = [_constant(gm, w13_node), _constant(gm, w2_node)]
        if any(t is None for t in packed):
            logger.warning("fused_moe: expert weights are not graph constants; leaving vLLM's op")
            continue
        isa = kw.get("isa", "amx")
        # Unpack real weights outside dynamo's active fake/functional modes.
        with _disable_current_modes():
            halves = split_gate_up(packed[0].detach(), isa)
            plain = None if halves is None else [*halves, unpack_expert_weight(packed[1].detach(), isa)]
        if plain is None or any(t is None for t in plain):
            logger.warning("fused_moe: expert weight packing for isa=%r is not a permutation", isa)
            continue

        sub = _trace_expert_graph(act, kw["input"].meta["val"].dtype, plain,
                                  kw["topk_weights"].meta["val"].dtype, top_ids.meta["val"])

        with gm.graph.inserting_before(node):
            attrs = []
            for tag, tensor in zip(("w1", "w3", "w2"), plain):
                name = f"_ov_moe{index}_{tag}"
                gm.register_buffer(name, tensor)
                attrs.append(gm.graph.get_attr(name))
            value_map = dict(zip((n for n in sub.graph.nodes if n.op == "placeholder"),
                                 (kw["input"], *attrs, kw["topk_weights"], top_ids)))
            result = gm.graph.graph_copy(sub.graph, value_map)

        replace_mutated_output(gm, node, result, kw.get("_output_base_index"))
        rewrites += 1

    if rewrites:
        gm.graph.lint()
        gm.recompile()
    return rewrites
