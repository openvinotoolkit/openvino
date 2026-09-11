# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""vLLM PagedAttention integration for the OV torchdynamo backend.

Registers a custom torch op `openvino::paged_attention(q, k, v, layer_name)`
whose Python impl delegates to vLLM's `unified_attention_with_output`, plus an
FX pre-pass that rewrites `auto_functionalized_v2(unified_attention_with_output)`
call sites into it. That turns attention from an untranslatable HOP into an
op the partitioner can keep inside an OV partition.

The C++ translator emits a PagedAttentionExtension for the op; side_channel.py
binds the KV cache, block tables and lengths from vllm.forward_context.
"""

# mypy: ignore-errors

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

_REGISTERED = False


def _register_custom_op():
    """Register torch.ops.openvino.paged_attention once per process."""
    global _REGISTERED
    if _REGISTERED:
        return
    import torch

    @torch.library.custom_op(
        "openvino::paged_attention",
        mutates_args=(),
    )
    def paged_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_name: str,
    ) -> torch.Tensor:
        # Only hit on the torch-eager fallback path. vLLM's CPU backend
        # implements just the "_with_output" variant, so pass in an output.
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
    ) -> torch.Tensor:
        return torch.empty_like(query).contiguous()

    _REGISTERED = True
    logger.debug("Registered torch.ops.openvino.paged_attention")


def _is_unified_attention_with_output(node) -> bool:
    """Match auto_functionalized_v2(unified_attention_with_output, ...)."""
    import torch
    if node.op != "call_function":
        return False
    tgt = node.target
    try:
        auto_fv2 = torch.ops.higher_order.auto_functionalized_v2
    except Exception:
        return False
    if tgt is not auto_fv2:
        return False
    if not node.args:
        return False
    inner = node.args[0]
    try:
        ua_overload = torch.ops.vllm.unified_attention_with_output.default
    except Exception:
        return False
    return inner is ua_overload


def rewrite_unified_attention_to_paged_attention(gm) -> int:
    """Rewrite every attention HOP node to a call of the OV paged_attention op.

    Matches auto_functionalized_v2(unified_attention_with_output, ...) nodes
    and rewrites them to torch.ops.openvino.paged_attention.default. Returns
    the number of rewrites performed.
    """
    import torch

    _register_custom_op()

    paged_attention_op = torch.ops.openvino.paged_attention.default

    # Collect first; do not mutate while iterating.
    to_rewrite = [n for n in gm.graph.nodes if _is_unified_attention_with_output(n)]
    if not to_rewrite:
        return 0

    rewrites = 0
    for node in to_rewrite:
        kw = dict(node.kwargs)
        q = kw.get("query")
        k = kw.get("key")
        v = kw.get("value")
        layer_name = kw.get("layer_name")
        if q is None or k is None or v is None or layer_name is None:
            logger.warning(
                "Skipping unified_attention rewrite: missing q/k/v/layer_name in kwargs"
            )
            continue

        with gm.graph.inserting_after(node):
            new_node = gm.graph.call_function(
                paged_attention_op,
                args=(q, k, v, layer_name),
            )

        # auto_functionalized_v2 returns (op_result, *bases_mutated_in_place),
        # and unified_attention_with_output writes its output to
        # _all_bases[_output_base_index]. So the one consumer that matters is
        # getitem(node, 1 + _output_base_index); our op returns that directly.
        output_base_index = kw.get("_output_base_index", 0)
        attn_out_getitem_idx = 1 + (output_base_index or 0)

        for user in list(node.users):
            if (
                user.op == "call_function"
                and user.target is __import__("operator").getitem
                and len(user.args) == 2
                and isinstance(user.args[1], int)
            ):
                if user.args[1] == attn_out_getitem_idx:
                    user.replace_all_uses_with(new_node)
                    gm.graph.erase_node(user)
                # Getitems of other indices (the op result at 0, other mutated
                # bases) stay wired to the original node, which then survives.

        if not node.users:
            gm.graph.erase_node(node)
        else:
            # Consumers remain for bases we did not rewrite, so the original
            # node stays and the partitioner splits here. Correct, just slower.
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
