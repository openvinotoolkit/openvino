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
        support_dict["torch.ops.openvino.paged_attention.default"] = None
        support_dict["torch.ops.openvino.gdn_attention.default"] = None


def maybe_rewrite_paged_attention(graph_module, options):
    """Rewrite vLLM's opaque attention, Gated DeltaNet and fused-MoE ops into OV-translatable ones.

    Each turns an auto_functionalized_v2 call site OV cannot hold into an op it can keep inside its
    partition: unified_attention_with_output -> openvino.paged_attention, cpu_gdn_attention_core ->
    openvino.gdn_attention ("gdn_translate"), and _C.cpu_fused_moe -> grouped matmuls ("moe_translate").
    All are off when "paged_attention" is. Returns the total number of rewrites; a failing rewrite
    leaves its call sites on vLLM's op.
    """
    from openvino.frontend.pytorch.torchdynamo.vllm.preset import bool_opt
    if not bool_opt(options, "paged_attention", True):
        return 0
    rewrites = 0
    try:
        from openvino.frontend.pytorch.torchdynamo.vllm.paged_attention import (
            rewrite_unified_attention_to_paged_attention,
        )
        rewrites += rewrite_unified_attention_to_paged_attention(graph_module)
    except Exception:
        pass
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
