# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""vLLM-specific glue for the OpenVINO torchdynamo backend.

Everything here is kept out of the generic torchdynamo backend so it stays
free of vLLM-specific knowledge. The entry point is plugin.register().
"""


def maybe_register_pa_op(support_dict, options):
    """Mark the OV paged_attention custom op supported.

    Keeps it inside the OV partition. No-op unless "pa_translate" is set.
    """
    from openvino.frontend.pytorch.torchdynamo.vllm.preset import bool_opt
    if bool_opt(options, "pa_translate", False):
        support_dict["torch.ops.openvino.paged_attention.default"] = None


def maybe_rewrite_paged_attention(graph_module, options):
    """Rewrite vLLM's attention HOP nodes into the OV paged_attention op.

    Turns auto_functionalized_v2(unified_attention_with_output) call sites
    into torch.ops.openvino.paged_attention.default, so OV can keep the call
    inside its partition. Returns the number of rewrites, or 0 on no-op /
    failure.
    """
    from openvino.frontend.pytorch.torchdynamo.vllm.preset import bool_opt
    if not bool_opt(options, "paged_attention", True):
        return 0
    try:
        from openvino.frontend.pytorch.torchdynamo.vllm.paged_attention import (
            rewrite_unified_attention_to_paged_attention,
        )
        return rewrite_unified_attention_to_paged_attention(graph_module)
    except Exception:
        return 0
