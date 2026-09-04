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

    model = torch.nn.Linear(64, 64).eval()
    compiled = torch.compile(model, backend="openvino", options={"aot_autograd": True})
    with torch.no_grad():
        compiled(torch.randn(1, 64))

    assert len(placeholders) == 1, (
        f"expected only the activation as a graph input, got {placeholders}; "
        "the weight and bias were not frozen into constants"
    )
