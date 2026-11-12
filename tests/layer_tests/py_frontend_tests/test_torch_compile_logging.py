# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest
import torch
from torch.fx import Graph, GraphModule, symbolic_trace

from openvino.frontend.pytorch.torchdynamo import backend
from openvino.frontend.pytorch.torchdynamo import execute as execute_module


@pytest.mark.precommit
def test_partition_logging_and_execution_fallback(caplog, monkeypatch):
    monkeypatch.setattr(execute_module, "max_openvino_partitions", 0)
    subgraph = symbolic_trace(lambda x: x + 1)
    graph = Graph()
    x = graph.placeholder("x")
    result = graph.call_module("fused_0", (x,))
    graph.output(result)
    model = GraphModule({"fused_0": subgraph}, graph)

    with caplog.at_level(logging.DEBUG, logger="openvino.frontend.pytorch.torchdynamo"):
        execute_module.partition_graph(model, True)
        partition = model.fused_0
        monkeypatch.setattr(execute_module, "openvino_execute", lambda *args, **kwargs: torch.tensor(4))
        assert partition(torch.tensor(1)).item() == 4

        def fail(*args, **kwargs):
            raise RuntimeError("conversion failed")

        monkeypatch.setattr(execute_module, "openvino_execute", fail)
        assert partition(torch.tensor(1)).item() == 2
        assert partition(torch.tensor(2)).item() == 3

    assert "OpenVINO partition" in caplog.text and "fused_0" in caplog.text
    assert "executed with OpenVINO" in caplog.text
    assert "falling back to native PyTorch: conversion failed" in caplog.text
    assert "uses native PyTorch after an earlier OpenVINO failure" in caplog.text


@pytest.mark.precommit
def test_no_partitions_and_backend_fallback_logging(caplog, monkeypatch):
    model = symbolic_trace(lambda x: x + 1)
    with caplog.at_level(logging.DEBUG, logger="openvino.frontend.pytorch.torchdynamo"):
        execute_module.partition_graph(model, True)
        monkeypatch.setattr(backend, "_get_model_caching", lambda options: (_ for _ in ()).throw(RuntimeError("failed")))
        monkeypatch.setattr(backend, "compile_fx", lambda graph, inputs: graph.forward)
        result = backend.fx_openvino(model, [torch.tensor(1)])
        assert result(torch.tensor(1)).item() == 2

    assert "No OpenVINO partitions" in caplog.text
    assert "falling back to PyTorch Inductor: failed" in caplog.text
