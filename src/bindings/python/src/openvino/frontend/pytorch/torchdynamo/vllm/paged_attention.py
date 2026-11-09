# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Rewrite vLLM's attention HOP into ov_ext::paged_attention, which the translator lowers to PagedAttentionExtension.

The op carries what the FX graph can't recover: `scale` (Gemma 4 uses 1.0, not 1/sqrt(head_dim); 0.0 means derive it)
and `kv_sharing_target` (vLLM hands KV-sharing layers raw k/v and skips their cache write, but PA always writes, so the
translator reuses the target's k/v instead; "" means not shared).
"""

# mypy: ignore-errors

import logging

from .fx_utils import is_auto_functionalized, replace_mutated_output

logger = logging.getLogger(__name__)

_REGISTERED = False


def _register_custom_op():
    """Register torch.ops.ov_ext.paged_attention once per process."""
    global _REGISTERED
    if _REGISTERED:
        return
    import torch

    @torch.library.custom_op(
        "ov_ext::paged_attention",
        mutates_args=(),
    )
    def paged_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_name: str,
        scale: float,
        kv_sharing_target: str,
    ) -> torch.Tensor:
        # Eager fallback only; vLLM's Attention layer applies scale and KV sharing itself here.
        out = torch.empty_like(query).contiguous()
        torch.ops.vllm.unified_attention_with_output(
            query, key, value, out, layer_name
        )
        return out

    @paged_attention.register_fake
    def _paged_attention_fake(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_name: str,
        scale: float,
        kv_sharing_target: str,
    ) -> torch.Tensor:
        return torch.empty_like(query).contiguous()

    _REGISTERED = True
    logger.debug("Registered torch.ops.ov_ext.paged_attention")


def _attention_layer_meta(layer_name):
    """Return a layer's (scale, kv_sharing_target); (0.0, "") lets the translator derive them."""
    try:
        from vllm.forward_context import get_forward_context

        layers = get_forward_context().no_compile_layers
        layer = layers.get(layer_name) if isinstance(layers, dict) else None
    except Exception as e:
        logger.debug("no forward context for %s: %s", layer_name, e)
        return 0.0, ""
    if layer is None:
        logger.warning(
            "paged_attention: layer %r not in no_compile_layers; the OV "
            "translator will derive the scale and assume no KV sharing",
            layer_name)
        return 0.0, ""
    # Attention.extra_repr treats impl.scale as the authoritative value.
    scale = getattr(getattr(layer, "impl", None), "scale", None)
    target = getattr(layer, "kv_sharing_target_layer_name", None)
    return (float(scale) if scale is not None else 0.0), str(target or "")


def _is_unified_attention_with_output(node) -> bool:
    """Match auto_functionalized_v2(unified_attention_with_output, ...)."""
    import torch
    return is_auto_functionalized(node, lambda: torch.ops.vllm.unified_attention_with_output.default)


def count_unified_attention(gm) -> int:
    """Count the vLLM attention call sites in gm."""
    return sum(1 for n in gm.graph.nodes if _is_unified_attention_with_output(n))


def rewrite_unified_attention_to_paged_attention(gm) -> int:
    """Rewrite every attention HOP node to ov_ext.paged_attention; returns the number of rewrites."""
    import torch

    _register_custom_op()

    paged_attention_op = torch.ops.ov_ext.paged_attention.default

    to_rewrite = [n for n in gm.graph.nodes if _is_unified_attention_with_output(n)]
    if not to_rewrite:
        return 0

    rewrites = 0
    for node in to_rewrite:
        kw = dict(node.kwargs)
        query = kw.get("query")
        key = kw.get("key")
        value = kw.get("value")
        layer_name = kw.get("layer_name")
        if query is None or key is None or value is None or layer_name is None:
            logger.warning(
                "Skipping unified_attention rewrite: missing q/k/v/layer_name in kwargs"
            )
            continue

        scale, kv_sharing_target = _attention_layer_meta(layer_name)
        with gm.graph.inserting_after(node):
            new_node = gm.graph.call_function(
                paged_attention_op,
                args=(query, key, value, layer_name, scale,
                      kv_sharing_target),
            )

        # auto_functionalized_v2 writes output to _all_bases[_output_base_index].
        if not replace_mutated_output(gm, node, new_node, kw.get("_output_base_index")):
            # Other consumers keep the original node; the partitioner splits here (correct, just slower).
            logger.debug(
                f"Left original auto_functionalized_v2 node for layer "
                f"{layer_name}: it still has {len(node.users)} non-attention "
                f"consumers"
            )

        rewrites += 1

    if rewrites:
        gm.graph.lint()
        gm.recompile()

    return rewrites
