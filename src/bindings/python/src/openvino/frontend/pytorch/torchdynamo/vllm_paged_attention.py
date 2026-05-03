# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""vLLM PagedAttention integration for the OV torchdynamo backend.

This module:
  1. Registers a custom torch op `openvino::paged_attention(q, k, v, layer_name)`
     that delegates at runtime to vLLM's `unified_attention_with_output`.
     Its Python impl is the torch-fallback path.
  2. Provides an FX pre-pass `rewrite_unified_attention_to_paged_attention(gm)`
     that replaces `auto_functionalized_v2(unified_attention_with_output, ...)`
     call sites with calls to the custom op above. This turns attention from
     an untranslatable HOP into an OV-translatable op, enabling the partitioner
     to keep the attention call inside an OV partition.

The OV frontend emits a PagedAttentionExtension for this op (see C++ translator)
and the execute-time binding in execute.py fills the KV cache / block tables /
past_lens etc. from vllm.forward_context.get_forward_context().
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

    # Define the op with a tensor-list-returning signature. We keep a simple
    # interface: (q, k, v, layer_name) -> output_tensor. The python impl calls
    # vLLM's own unified_attention so torch-eager fallback stays correct.
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
        # Delegate to vLLM. This is only hit on the torch-eager fallback path;
        # the OV partition uses a C++ translator to emit PagedAttentionExtension.
        # vLLM's CPU backend only implements the "_with_output" path, so allocate
        # an output tensor and pass it in.
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
        # Output has same leading dim as q (num_tokens), hidden size = q heads*head_dim
        return torch.empty_like(query).contiguous()

    _REGISTERED = True
    logger.debug("Registered torch.ops.openvino.paged_attention")


def _dump_all_auto_functionalized_targets(gm):
    import torch
    try:
        auto_fv2 = torch.ops.higher_order.auto_functionalized_v2
    except Exception:
        return
    import os
    if not os.environ.get("OV_DBG_PA_TARGETS"):
        return
    seen = {}
    for n in gm.graph.nodes:
        if n.op == "call_function" and n.target is auto_fv2 and n.args:
            t = str(n.args[0])
            seen[t] = seen.get(t, 0) + 1
    if seen:
        print(f"[PA_TARGETS] {seen}", flush=True)


def _is_unified_attention_with_output(node) -> bool:
    """Match auto_functionalized_v2(unified_attention_with_output, ...)."""
    import torch
    if node.op != "call_function":
        return False
    tgt = node.target
    # Check the higher-order op wrapper
    try:
        auto_fv2 = torch.ops.higher_order.auto_functionalized_v2
    except Exception:
        return False
    if tgt is not auto_fv2:
        return False
    if not node.args:
        return False
    inner = node.args[0]
    # inner is a torch.ops.vllm.unified_attention_with_output.default OpOverload
    try:
        ua_overload = torch.ops.vllm.unified_attention_with_output.default
    except Exception:
        return False
    return inner is ua_overload


def rewrite_unified_attention_to_paged_attention(gm) -> int:
    """Rewrite every auto_functionalized_v2(unified_attention_with_output, ...)
    node to a call of torch.ops.openvino.paged_attention.default.

    Returns the number of rewrites performed.
    """
    import torch

    _register_custom_op()

    paged_attention_op = torch.ops.openvino.paged_attention.default

    _dump_all_auto_functionalized_targets(gm)

    # Collect candidate nodes first (don't mutate while iterating)
    to_rewrite = [n for n in gm.graph.nodes if _is_unified_attention_with_output(n)]
    if not to_rewrite:
        return 0

    import os as _os
    _dbg = _os.environ.get("OV_DBG_PA")
    rewrites = 0
    for node in to_rewrite:
        if _dbg:
            kw_keys = list(node.kwargs.keys())
            user_info = []
            for u in node.users:
                if u.op == "call_function" and u.target is __import__("operator").getitem:
                    user_info.append(f"getitem[{u.args[1]}]")
                else:
                    user_info.append(f"{u.op}:{u.target}")
            print(f"[PA_NODE] layer={node.kwargs.get('layer_name')} kwargs={kw_keys} users={user_info}", flush=True)
        # node.kwargs has: query, key, value, layer_name, output_scale,
        # kv_cache_dummy_dep, _output_base_index, _output_size, _output_stride,
        # _output_storage_offset, _output_block_scale_base_index, _all_bases
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

        # auto_functionalized_v2 returns a tuple; conventionally (None, *new_bases)
        # where new_bases is a copy of _all_bases mutated in place. Consumers of
        # node typically getitem(node, idx) to pull out the updated base.
        # Our custom op returns a single tensor (the attention output). We need
        # to replace every `getitem(node, <any idx>)` consumer with our result.

        # Insert new node just after the original
        with gm.graph.inserting_after(node):
            new_node = gm.graph.call_function(
                paged_attention_op,
                args=(q, k, v, layer_name),
            )

        # Find all getitem consumers of the original node
        # For auto_functionalized_v2, index 0 is the original op return (often
        # None for ops that return via mutation), index 1..N are the base
        # tensors mutated in place. For unified_attention_with_output, the
        # attention output is written to _all_bases[_output_base_index], so the
        # consumer that matters is getitem(node, 1 + _output_base_index).
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
                    # Replace this getitem's uses with new_node
                    user.replace_all_uses_with(new_node)
                    gm.graph.erase_node(user)
                else:
                    # Other getitems (e.g., the None at idx 0, or other bases)
                    # — pass them through as best-effort. For idx 0 (None), we
                    # leave it; for other bases we leave them wired to the
                    # original node but strip the op — can't, the node stays.
                    # Best simplification: if there are no other consumers of
                    # unrelated indices, we can fully replace.
                    pass

        # If the original node now has no users, erase it
        if not node.users:
            gm.graph.erase_node(node)
        else:
            # Still has consumers for other bases we didn't rewrite; leave it.
            # The partitioner will split here, which is OK — it's a correctness
            # fallback.
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
