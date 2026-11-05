# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Verify a rewritten attention call always runs inside an OV partition when OV owns the KV-cache write.

The partitioner drops single-node partitions outside testing, so an attention call between two ops OV doesn't
support would run its eager fallback, vLLM's attention op, which on this path reads a cache nobody writes. The
call must get a partition of its own, and a call OV can't take at all must fail the compilation.

No model download needed: a hand-built FX graph around the ov_ext.paged_attention op.
"""

import pytest
import torch

from openvino.frontend.pytorch.torchdynamo.partition import Partitioner
from openvino.frontend.pytorch.torchdynamo.vllm.paged_attention import _register_custom_op

_ISOLATING_OPS = ["torch.ops.aten.sin.default", "torch.ops.aten.cos.default"]


def _attention_between_unsupported_ops():
    """sin -> paged_attention -> cos, with sin and cos disabled below."""
    _register_custom_op()
    graph = torch.fx.Graph()
    x, k, v = (graph.placeholder(name) for name in ("x", "k", "v"))
    q = graph.call_function(torch.ops.aten.sin.default, (x,))
    out = graph.call_function(torch.ops.ov_ext.paged_attention.default, (q, k, v, "layers.0.attn", 0.125, ""))
    graph.output(graph.call_function(torch.ops.aten.cos.default, (out,)))
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def _top_level_attention(gm):
    op = torch.ops.ov_ext.paged_attention.default
    return [n for n in gm.graph.nodes if n.op == "call_function" and n.target is op]


@pytest.mark.precommit
def test_isolated_attention_gets_its_own_partition():
    options = {"vllm": True, "disabled_ops": _ISOLATING_OPS}
    fused = Partitioner(options).make_partitions(_attention_between_unsupported_ops(), options)
    assert _top_level_attention(fused) == [], f"attention left outside OV partitions:\n{fused.graph}"
    fused_modules = [n for n in fused.graph.nodes if n.op == "call_module" and n.target.startswith("fused_")]
    assert len(fused_modules) == 1


@pytest.mark.precommit
def test_attention_ov_cannot_take_raises():
    options = {"vllm": True, "disabled_ops": _ISOLATING_OPS + ["torch.ops.ov_ext.paged_attention.default"]}
    with pytest.raises(RuntimeError, match="outside an OpenVINO partition"):
        Partitioner(options).make_partitions(_attention_between_unsupported_ops(), options)


@pytest.mark.precommit
def test_isolated_attention_left_alone_without_kv_ownership():
    """Without the vLLM preset the KV write stays in vLLM's op, so the eager fallback is correct."""
    options = {"disabled_ops": _ISOLATING_OPS, "pa_translate": True}
    fused = Partitioner(options).make_partitions(_attention_between_unsupported_ops(), options)
    assert len(_top_level_attention(fused)) == 1
