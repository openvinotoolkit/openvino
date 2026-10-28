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
    if owns_kv_cache_update(options):
        rewrites += rewrite_unified_attention_to_paged_attention(graph_module)
        left = count_unified_attention(graph_module)
        if left:
            raise RuntimeError(
                f"OpenVINO vLLM backend: {left} attention call site(s) could not be rewritten to "
                "ov_ext.paged_attention. The KV-cache write is owned by the attention op on this path, "
                "so running them on vLLM's op would decode against an unwritten cache.")
    elif is_vllm_preset(options):
        rewrites += rewrite_unified_attention_to_paged_attention(graph_module)
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
