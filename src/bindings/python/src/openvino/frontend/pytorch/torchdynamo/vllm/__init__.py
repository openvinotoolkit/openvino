# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# mypy: ignore-errors

"""vLLM-specific glue for the OpenVINO torchdynamo backend.

Everything here is kept out of the generic torchdynamo backend so it stays
free of vLLM-specific knowledge. The entry point is plugin.register().
"""

import logging

logger = logging.getLogger(__name__)


def maybe_register_pa_op(support_dict, options):
    """Mark the OV paged_attention and gdn_attention custom ops supported.

    Keeps them inside the OV partition. No-op unless "pa_translate" is set.
    """
    from openvino.frontend.pytorch.torchdynamo.vllm.preset import bool_opt
    if bool_opt(options, "pa_translate", False):
        support_dict["torch.ops.ov_ext.paged_attention.default"] = None
        support_dict["torch.ops.ov_ext.gdn_attention.default"] = None


def owns_kv_cache_update(options) -> bool:
    """True if the KV-cache write belongs to the OV attention op: the rewrite is on and OV runs it."""
    from openvino.frontend.pytorch.torchdynamo.vllm.preset import bool_opt, is_vllm_preset
    return (is_vllm_preset(options) and bool_opt(options, "paged_attention", True)
            and bool_opt(options, "pa_translate", False))


def keep_paged_attention_partitioned(graph_module, partitions, supported_ops, options) -> int:
    """Give each ov_ext.paged_attention call left outside the proposed partitions a partition of its own.

    When OV owns the KV-cache write, the call's eager fallback (vLLM's attention op) reads a cache nobody
    writes, so the call must run in OV. The partitioner drops single-node partitions outside testing, which
    leaves an attention call between two unsupported ops on that fallback. A call OV doesn't support at all
    raises. Appends to `partitions` and returns the number of partitions added.

    A layer that shares another layer's KV cache is translated with its target's K/V, which the frontend
    finds only when both are converted together; in different partitions (or graphs) it would write its raw
    K/V into the shared cache. Such a split raises too.
    """
    if not owns_kv_cache_update(options):
        return 0
    import torch
    from torch.fx.passes.infra.partitioner import Partition
    from openvino.frontend.pytorch.torchdynamo.vllm.paged_attention import _register_custom_op

    _register_custom_op()
    paged_attention_op = torch.ops.ov_ext.paged_attention.default
    assigned = {node for partition in partitions for node in partition.nodes}
    next_id = max((p.id for p in partitions if p.id is not None), default=-1) + 1
    submodules = dict(graph_module.named_modules())
    added = 0
    for node in graph_module.graph.nodes:
        if node.op != "call_function" or node.target is not paged_attention_op or node in assigned:
            continue
        if not supported_ops.is_node_supported(submodules, node):
            raise RuntimeError(
                f"OpenVINO vLLM backend: attention call {node.name} is outside an OpenVINO partition. The "
                "KV-cache write is owned by the attention op on this path, so vLLM's op would decode against "
                "an unwritten cache.")
        partitions.append(Partition(id=next_id + added, nodes=[node]))
        added += 1

    owner = {node: partition for partition in partitions for node in partition.nodes}
    attention = [n for n in graph_module.graph.nodes if n.op == "call_function" and n.target is paged_attention_op]
    by_layer = {n.args[3]: n for n in attention}
    for node in attention:
        target_layer = node.args[5] if len(node.args) > 5 else ""
        if target_layer and owner.get(by_layer.get(target_layer)) is not owner[node]:
            raise RuntimeError(
                f"OpenVINO vLLM backend: attention layer {node.args[3]} shares the KV cache of {target_layer}, "
                "which is not in the same OpenVINO partition, so its own key/value would overwrite the shared "
                "cache.")
    return added


def maybe_rewrite_paged_attention(graph_module, options):
    """Rewrite vLLM's opaque attention, Gated DeltaNet and fused-MoE ops into OV-translatable ones.

    Each turns an auto_functionalized_v2 call site OV cannot hold into an op it can keep inside its
    partition: unified_attention_with_output -> ov_ext.paged_attention, cpu_gdn_attention_core ->
    ov_ext.gdn_attention ("gdn_translate"), and _C.cpu_fused_moe -> a tiled MoE block ("moe_translate").
    All are off when "paged_attention" is. Returns the total number of rewrites; a failing GDN or MoE
    rewrite leaves its call sites on vLLM's op.

    On the vLLM path the attention rewrite is mandatory: the plugin has moved the KV-cache write
    into the attention op (see plugin._own_kv_cache_update), and vLLM's own attention op doesn't
    write it. So any attention call site left on vLLM's op raises instead of decoding against a
    cache nobody updates.
    """
    from openvino.frontend.pytorch.torchdynamo.vllm.preset import bool_opt, is_vllm_preset
    if not bool_opt(options, "paged_attention", True):
        return 0
    from openvino.frontend.pytorch.torchdynamo.vllm.paged_attention import (
        count_unified_attention,
        rewrite_unified_attention_to_paged_attention,
    )
    rewrites = 0
    if is_vllm_preset(options):
        rewrites += rewrite_unified_attention_to_paged_attention(graph_module)
        if owns_kv_cache_update(options) and (left := count_unified_attention(graph_module)):
            raise RuntimeError(
                f"OpenVINO vLLM backend: {left} attention call site(s) could not be rewritten to "
                "ov_ext.paged_attention. The KV-cache write is owned by the attention op on this path, "
                "so running them on vLLM's op would decode against an unwritten cache.")
    else:
        try:
            rewrites += rewrite_unified_attention_to_paged_attention(graph_module)
        except Exception as e:
            logger.debug("paged_attention rewrite skipped: %s", e)
    if bool_opt(options, "gdn_translate", True):
        try:
            from openvino.frontend.pytorch.torchdynamo.vllm.gated_delta_net import rewrite_gdn_to_ov
            rewrites += rewrite_gdn_to_ov(graph_module)
        except Exception as e:
            logger.warning("gdn_attention rewrite skipped: %s", e)
    if bool_opt(options, "moe_translate", True):
        try:
            from openvino.frontend.pytorch.torchdynamo.vllm.fused_moe import rewrite_cpu_fused_moe
            rewrites += rewrite_cpu_fused_moe(graph_module)
        except Exception as e:
            logger.warning("fused_moe rewrite skipped: %s", e)
    return rewrites
