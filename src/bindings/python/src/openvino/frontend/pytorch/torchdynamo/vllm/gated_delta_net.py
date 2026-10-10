# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Rewrites vLLM's cpu_gdn_attention_core into openvino::gdn_attention (lowered to PagedGatedDeltaNet)."""

# mypy: ignore-errors

import logging

from .fx_utils import is_auto_functionalized, packing_permutation, replace_mutated_output

logger = logging.getLogger(__name__)

_REGISTERED = False
_conv_perm_cache = {}


def _register_custom_op():
    global _REGISTERED
    if _REGISTERED:
        return
    from typing import Optional

    import torch

    @torch.library.custom_op("openvino::gdn_attention", mutates_args=())
    def gdn_attention(
        mixed_qkv: torch.Tensor,
        b_proj: torch.Tensor,
        a_proj: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: Optional[torch.Tensor],
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        layer_name: str,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
    ) -> torch.Tensor:
        # Eager fallback (profile run, or a partition OV rejected): vLLM's own op.
        out = torch.zeros((mixed_qkv.shape[0], num_v_heads, head_v_dim),
                          dtype=mixed_qkv.dtype, device=mixed_qkv.device)
        torch.ops.vllm.cpu_gdn_attention_core(mixed_qkv, b_proj, a_proj, out, layer_name)
        return out

    @gdn_attention.register_fake
    def _gdn_attention_fake(mixed_qkv, b_proj, a_proj, conv_weight, conv_bias, A_log, dt_bias,
                            layer_name, num_k_heads, num_v_heads, head_k_dim, head_v_dim):
        return mixed_qkv.new_empty((mixed_qkv.shape[0], num_v_heads, head_v_dim))

    _REGISTERED = True


def _plain_conv_weight(layer):
    """Conv weight as plain [channels, kernel], undoing vLLM's AMX prepack; None if it can't be undone."""
    import torch
    channels = layer.conv_dim
    weight = layer.conv1d.weight.detach().reshape(channels, -1)
    if not torch.cpu._is_amx_tile_supported():
        return weight
    import vllm._custom_ops as ops
    key = (tuple(weight.shape), weight.dtype)
    if key not in _conv_perm_cache:
        _conv_perm_cache[key] = packing_permutation(ops.causal_conv1d_weight_pack, weight.shape, weight.dtype)
    source = _conv_perm_cache[key]
    if source is None:
        return None
    plain = torch.empty_like(weight).reshape(-1)
    plain[source] = weight.reshape(-1)
    return plain.reshape(weight.shape)


def _gdn_core_op():
    import torch
    return torch.ops.vllm.cpu_gdn_attention_core.default


def _gdn_layer(layer_name):
    from vllm.forward_context import get_forward_context
    layers = get_forward_context().no_compile_layers
    return layers.get(layer_name) if isinstance(layers, dict) else None


def rewrite_gdn_to_ov(gm) -> int:
    """Rewrite vLLM's CPU GDN core calls into openvino.gdn_attention; unresolvable sites stay on vLLM's op."""
    import torch
    from torch.utils._python_dispatch import _disable_current_modes

    nodes = [n for n in gm.graph.nodes if is_auto_functionalized(n, _gdn_core_op)]
    if not nodes:
        return 0
    _register_custom_op()
    gdn_op = torch.ops.openvino.gdn_attention.default

    rewrites = 0
    for i, node in enumerate(nodes):
        kw = dict(node.kwargs)
        layer_name = kw.get("layer_name")
        layer = _gdn_layer(layer_name) if isinstance(layer_name, str) else None
        if layer is None or getattr(layer, "tp_size", 1) != 1:
            logger.warning("gdn_attention: leaving %r on vLLM's op (layer unresolved or TP>1)", layer_name)
            continue

        # Real weights; the rewrite runs under dynamo's fake/functional modes.
        with _disable_current_modes():
            conv_weight = _plain_conv_weight(layer)
            if conv_weight is None:
                logger.warning("gdn_attention: leaving %r on vLLM's op (conv weight packing not a permutation)",
                               layer_name)
                continue
            weights = {
                "conv_weight": conv_weight,
                "A_log": layer.A_log.detach(),
                "dt_bias": layer.dt_bias.detach(),
            }
            if layer.conv1d.bias is not None:
                weights["conv_bias"] = layer.conv1d.bias.detach()
        attr_nodes = {}
        with gm.graph.inserting_before(node):
            for wname, tensor in weights.items():
                attr = f"_ov_gdn{i}_{wname}"
                gm.register_buffer(attr, tensor)
                attr_nodes[wname] = gm.graph.get_attr(attr)
            new_node = gm.graph.call_function(
                gdn_op,
                args=(kw["mixed_qkv"], kw["b"], kw["a"], attr_nodes["conv_weight"],
                      attr_nodes.get("conv_bias"), attr_nodes["A_log"], attr_nodes["dt_bias"],
                      layer_name, layer.num_k_heads, layer.num_v_heads,
                      layer.head_k_dim, layer.head_v_dim),
            )

        # The core's output is the mutated base _all_bases[_core_attn_out_base_index].
        replace_mutated_output(gm, node, new_node, kw.get("_core_attn_out_base_index"))
        rewrites += 1

    if rewrites:
        gm.graph.lint()
        gm.recompile()
    return rewrites
