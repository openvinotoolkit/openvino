# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""vLLM-specific glue for the OpenVINO torchdynamo backend.

Modules:
    plugin           - vLLM general_plugins entry point; wires the OV backend
                       into vllm.v1.worker.cpu_model_runner.
    paged_attention  - Custom torch op + FX rewrite that converts vLLMs
                       unified_attention_with_output HOP into a flat OV PA op.
    sampler          - OV-compiled fast path for vllm.v1.sample.sampler.Sampler,
                       gated by an eligibility check.
"""


def maybe_register_pa_op(support_dict, options):
    """If options["pa_translate"] is set, register the OV paged_attention
    custom op as supported in the OperatorSupport dict so the OV partitioner
    keeps it inside its partition."""
    from openvino.frontend.pytorch.torchdynamo.backend_utils import _bool_opt
    if _bool_opt(options, "pa_translate", False):
        support_dict["torch.ops.openvino.paged_attention.default"] = None


def maybe_rewrite_paged_attention(graph_module, options):
    """If options["paged_attention"] is set, rewrite vLLM's
    auto_functionalized_v2(unified_attention_with_output, ...) HOP nodes into
    torch.ops.openvino.paged_attention.default so OV can keep the call inside
    its partition. No-op on graphs that have no such nodes.

    Returns the number of rewrites performed, or 0 on no-op / failure.
    """
    from openvino.frontend.pytorch.torchdynamo.backend_utils import _bool_opt
    if not _bool_opt(options, "paged_attention", True):
        return 0
    try:
        from openvino.frontend.pytorch.torchdynamo.vllm.paged_attention import (
            rewrite_unified_attention_to_paged_attention,
        )
        return rewrite_unified_attention_to_paged_attention(graph_module)
    except Exception:
        return 0
