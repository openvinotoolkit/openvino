# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from openvino.frontend.pytorch.torchdynamo import execute


@pytest.mark.nightly
@pytest.mark.precommit_fx_backend
def test_module_weights_are_frozen_into_constants(ie_device, precision, monkeypatch):
    placeholders = []
    original_partitioner = execute.partition_graph

    def spy(graph_module, *args, **kwargs):
        if not placeholders:
            placeholders.extend(
                node.name for node in graph_module.graph.nodes if node.op == "placeholder"
            )
        return original_partitioner(graph_module, *args, **kwargs)

    monkeypatch.setattr(execute, "partition_graph", spy)

    class Int4Linear(torch.nn.Module):
        def __init__(self, out_features, in_features):
            super().__init__()
            self.register_buffer(
                "packed_weight",
                torch.randint(0, 256, (out_features, in_features // 2), dtype=torch.uint8),
            )
            self.register_buffer("scale", torch.rand(out_features, 1))

        def forward(self, x):
            low = torch.bitwise_and(self.packed_weight, 0xF)
            high = torch.bitwise_right_shift(self.packed_weight, 4)
            weight = torch.stack((low, high), dim=-1).reshape(self.packed_weight.shape[0], -1)
            return x @ (weight.to(torch.float32) * self.scale).T

    model = Int4Linear(64, 64).eval()
    compiled = torch.compile(model, backend="openvino", options={"aot_autograd": True})
    with torch.no_grad():
        compiled(torch.randn(1, 64))

    assert len(placeholders) == 1, (
        f"expected only the activation as a graph input, got {placeholders}; "
        "the packed weights and scales were not frozen into constants"
    )
