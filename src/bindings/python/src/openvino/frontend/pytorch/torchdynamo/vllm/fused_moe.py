# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Replace vLLM's opaque _C.cpu_fused_moe with the tiled MoE block the CPU plugin lowers to GatherMatmul.

If ConvertTiledMoeBlockTo3GatherMatmuls doesn't match, the block runs as dense batched matmuls over all experts.
NNCF int4 experts (quantization.NNCFFusedMoEMethod) skip vLLM's op and trace the same block via nncf_moe_forward.
"""


import logging
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .fx_utils import is_auto_functionalized, packing_permutation, replace_mutated_output

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

_ACTIVATIONS = ("silu", "gelu", "gelu_tanh")
_perm_cache: "dict[tuple[int, int, str], torch.Tensor | None]" = {}
# Unpacked weights per packed storage, shared by every compiled graph of the model.
_unpacked_cache: "weakref.WeakValueDictionary[tuple[int, tuple[int, ...], str], torch.Tensor]" = (
    weakref.WeakValueDictionary())
# Gate and up halves of an unpacked w13, keyed like _unpacked_cache plus the half.
_half_cache: "weakref.WeakValueDictionary[tuple[int, tuple[int, ...], str, int], torch.Tensor]" = (
    weakref.WeakValueDictionary())
# Traced expert graphs per layer signature; every MoE layer of a model shares one.
_trace_cache: "dict[tuple[Any, ...], torch.fx.GraphModule]" = {}


def clear_weight_caches() -> None:
    """Drop the unpacked expert copies: their packed-address keys survive in-place reloads."""
    _unpacked_cache.clear()
    _half_cache.clear()


def _unpack_permutation(rows: int, cols: int, isa: str) -> "torch.Tensor | None":
    """Source index per packed position of a [rows, cols] expert weight; None if the pack isn't a permutation."""
    import torch
    import vllm._custom_ops as ops
    key = (rows, cols, isa)
    if key not in _perm_cache:
        _perm_cache[key] = packing_permutation(lambda t: ops.cpu_prepack_moe_weight(t, isa),
                                               (1, rows, cols), torch.bfloat16)
    return _perm_cache[key]


def unpack_expert_weight(packed: "torch.Tensor", isa: str) -> "torch.Tensor | None":
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


def split_gate_up(packed: "torch.Tensor", isa: str) -> "list[torch.Tensor] | None":
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


def moe_tiled(tokens: "torch.Tensor", w1: "torch.Tensor", w3: "torch.Tensor", w2: "torch.Tensor",
              topk_weights: "torch.Tensor", topk_ids: "torch.Tensor", act: str) -> "torch.Tensor":
    """Compute MoE experts as the tiled block ConvertTiledMoeBlockTo3GatherMatmuls matches (w1 = gate, w3 = up).

    bmm and view keep -1 for the token dim in the trace, where matmul and reshape would bake it in.
    """
    import torch
    experts, hidden = w1.shape[0], tokens.shape[1]
    tiled = tokens.repeat(experts, 1).view(experts, -1, hidden)
    gate = torch.bmm(tiled, w1.transpose(1, 2))
    up = torch.bmm(tiled, w3.transpose(1, 2))
    gated = _activation(gate, act) * up
    down = torch.bmm(gated, w2.transpose(1, 2)).view(experts, -1, hidden)
    routing = torch.zeros_like(tokens[:, :1]).expand(-1, experts)
    routing = routing.scatter(1, topk_ids, topk_weights.to(tokens.dtype))
    routing = routing.transpose(0, 1).contiguous().view(experts, -1, 1)
    return (down * routing).sum(0)


def _activation(gate: "torch.Tensor", act: str) -> "torch.Tensor":
    from torch.nn import functional
    if act == "silu":
        return functional.silu(gate)
    return functional.gelu(gate, approximate="tanh" if act == "gelu_tanh" else "none")


def nncf_moe_forward(tokens: "torch.Tensor", weights: "list[list[torch.Tensor]]", topk_weights: "torch.Tensor",
                     topk_ids: "torch.Tensor", act: str,
                     dequantize: "Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]") -> "torch.Tensor":
    """MoE over NNCF-compressed experts; `weights` is [qweight, scale, zero_point] for w1 (gate), w3 (up), w2.

    Traced, it is moe_tiled over the decompression chains, which the CPU plugin folds into GatherMatmulCompressed.
    Eager, only the routed experts are decompressed.
    """
    import torch
    from torch.nn import functional
    if torch.compiler.is_compiling():
        w1, w3, w2 = (dequantize(*w) for w in weights)
        return moe_tiled(tokens, w1, w3, w2, topk_weights, topk_ids, act)
    out = torch.zeros_like(tokens)
    for expert in topk_ids.unique().tolist():
        rows, slots = (topk_ids == expert).nonzero(as_tuple=True)
        w1, w3, w2 = (dequantize(*(t[expert] for t in w)) for w in weights)
        x = tokens[rows]
        down = functional.linear(_activation(functional.linear(x, w1), act) * functional.linear(x, w3), w2)
        out.index_add_(0, rows, down * topk_weights[rows, slots, None].to(tokens.dtype))
    return out


def _constant(gm: "torch.fx.GraphModule", node: "torch.fx.Node") -> "torch.Tensor | None":
    """Return the tensor behind a get_attr node, or None."""
    if node.op != "get_attr":
        return None
    value = gm
    for part in node.target.split("."):
        value = getattr(value, part)
    return value


def _fused_moe_op() -> "torch._ops.OpOverload":
    import torch
    return torch.ops._C.cpu_fused_moe.default


def _trace_expert_graph(act: str, x_dtype: "torch.dtype", weights: "list[torch.Tensor]", w_dtype: "torch.dtype",
                        top_ids_val: "torch.Tensor") -> "torch.fx.GraphModule":
    """Return the FX graph of moe_tiled for one layer signature, traced once."""
    import torch
    from torch._guards import tracing
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch.utils._python_dispatch import _disable_current_modes

    w1, w3, w2 = weights
    top_k, id_dtype = top_ids_val.shape[1], top_ids_val.dtype
    key = (act, x_dtype, tuple(w1.shape), tuple(w2.shape), w1.dtype, w_dtype, top_k, id_dtype)
    if key not in _trace_cache:
        def body(tokens: "torch.Tensor", w1: "torch.Tensor", w3: "torch.Tensor", w2: "torch.Tensor",
                 top_w: "torch.Tensor", top_i: "torch.Tensor") -> "torch.Tensor":
            return moe_tiled(tokens, w1, w3, w2, top_w, top_i, act)

        # Fakes from a private static-shape mode, so dynamo's ShapeEnv gets no guards.
        with tracing(None), _disable_current_modes():
            fake_mode = FakeTensorMode(static_shapes=True)
            examples = [fake_mode.from_tensor(t, static_shapes=True) for t in (
                torch.empty(4, w1.shape[2], dtype=x_dtype), w1, w3, w2,
                torch.empty(4, top_k, dtype=w_dtype), torch.empty(4, top_k, dtype=id_dtype))]
            _trace_cache[key] = make_fx(body, tracing_mode="fake")(*examples)
    return _trace_cache[key]


def rewrite_cpu_fused_moe(gm: "torch.fx.GraphModule") -> int:
    """Replace _C.cpu_fused_moe call sites with the tiled MoE formulation."""
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
        assert packed[0] is not None
        assert packed[1] is not None
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
