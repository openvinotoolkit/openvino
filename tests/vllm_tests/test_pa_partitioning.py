# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Verify a rewritten attention call always runs inside an OV partition when OV owns the KV-cache write.

The partitioner drops single-node partitions outside testing, so an attention call between two ops OV doesn't
support would run its eager fallback, vLLM's attention op, which on this path reads a cache nobody writes. The
call must get a partition of its own, and a call OV can't take at all must fail the compilation.

A layer that shares another layer's KV cache is translated with its target's K/V, which the frontend only finds
when both are converted together; split across partitions it would overwrite the shared cache with its raw K/V,
so that split must fail the compilation too.

No model download needed: hand-built FX graphs around the ov_ext.paged_attention op.
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


def _kv_sharing_pair(split):
    """Layer 1 shares layer 0's KV cache; with `split`, a disabled sin sits between the two attention calls."""
    _register_custom_op()
    graph = torch.fx.Graph()
    x, k, v = (graph.placeholder(name) for name in ("x", "k", "v"))
    pa = torch.ops.ov_ext.paged_attention.default
    out0 = graph.call_function(pa, (x, k, v, "layers.0.attn", 0.125, ""))
    q1 = graph.call_function(torch.ops.aten.sin.default if split else torch.ops.aten.neg.default, (out0,))
    out1 = graph.call_function(pa, (q1, k, v, "layers.1.attn", 0.125, "layers.0.attn"))
    graph.output(out1)
    return torch.fx.GraphModule(torch.nn.Module(), graph)


@pytest.mark.precommit
def test_kv_sharing_split_from_its_target_raises():
    options = {"vllm": True, "disabled_ops": ["torch.ops.aten.sin.default"]}
    with pytest.raises(RuntimeError, match="shares the KV cache of layers.0.attn"):
        Partitioner(options).make_partitions(_kv_sharing_pair(split=True), options)


@pytest.mark.precommit
def test_kv_sharing_with_its_target_in_one_partition():
    options = {"vllm": True, "disabled_ops": ["torch.ops.aten.sin.default"]}
    fused = Partitioner(options).make_partitions(_kv_sharing_pair(split=False), options)
    assert _top_level_attention(fused) == []
    assert len([n for n in fused.graph.nodes if n.op == "call_module"]) == 1
