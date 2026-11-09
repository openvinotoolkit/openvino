# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""runtime_hooks.infer_with_pa (OV_FAST_INFER) binds bf16 inputs zero-copy and runs inference at most once per call.

A rerun after req.infer() would advance the stateful ops (KV cache, GDN state) twice for one step.
"""

import numpy as np
import openvino as ov
import openvino.opset13 as ops
import pytest
import torch

from openvino.frontend.pytorch.torchdynamo.execute import _ov_result_to_torch, _torch_to_numpy
from openvino.frontend.pytorch.torchdynamo.vllm import runtime_hooks


def _compile(element_type, monkeypatch):
    monkeypatch.setenv("OV_FAST_INFER", "1")
    param = ops.parameter([2, 4], element_type, name="x")
    return ov.Core().compile_model(ov.Model([ops.multiply(param, ops.constant(2.0, element_type))], [param]), "CPU")


@pytest.fixture(autouse=True)
def _clear_caches():
    runtime_hooks.clear_caches()
    yield
    runtime_hooks.clear_caches()


class _Request:
    """Forward to a real InferRequest, counting infer() runs and failing where asked."""

    def __init__(self, req, fail_bind=False, fail_output=False):
        self._req = req
        self._fail_bind = fail_bind
        self._fail_output = fail_output
        self.runs = 0

    def set_tensor(self, port, tensor):
        if self._fail_bind:
            raise RuntimeError("injected binding failure")
        self._req.set_tensor(port, tensor)

    def infer(self, *args, **kwargs):
        self.runs += 1
        return self._req.infer(*args, **kwargs)

    def get_tensor(self, port):
        if self._fail_output:
            raise RuntimeError("injected output failure")
        return self._req.get_tensor(port)


@pytest.mark.precommit
def test_output_failure_does_not_rerun_inference(monkeypatch):
    compiled = _compile(ov.Type.f32, monkeypatch)
    req = _Request(compiled.create_infer_request(), fail_output=True)
    with pytest.raises(RuntimeError, match="injected output failure"):
        runtime_hooks.infer_with_pa(req, compiled, {compiled.inputs[0]: ov.Tensor(ov.Type.f32, [2, 4])})
    assert req.runs == 1, f"inference ran {req.runs} times for one call"


@pytest.mark.precommit
def test_binding_failure_falls_back_once(monkeypatch):
    compiled = _compile(ov.Type.f32, monkeypatch)
    req = _Request(compiled.create_infer_request(), fail_bind=True)
    runtime_hooks.infer_with_pa(req, compiled, {compiled.inputs[0]: ov.Tensor(ov.Type.f32, [2, 4])})
    assert req.runs == 1, f"inference ran {req.runs} times for one call"
    assert id(req) not in runtime_hooks._fastinfer_bound_ids


@pytest.mark.precommit
def test_fast_infer_binds_bf16_input(monkeypatch):
    """_torch_to_numpy hands bf16 over as same-bits float16; it must bind to the bf16 port, not fall back."""
    compiled = _compile(ov.Type.bf16, monkeypatch)
    req = compiled.create_infer_request()
    port = compiled.inputs[0]
    x = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    arr = _torch_to_numpy(x)
    assert arr.dtype == np.float16
    for _ in range(2):
        out = runtime_hooks.infer_with_pa(req, compiled, {port: arr})
        assert id(req) in runtime_hooks._fastinfer_bound_ids, "bf16 input fell back to req.infer"
        assert torch.equal(_ov_result_to_torch(out, compiled.outputs[0]), x * 2)
    bound = req.get_tensor(port)
    assert bound.element_type == ov.Type.bf16
    assert bound.data.ctypes.data == arr.ctypes.data, "bf16 input was copied, not shared"
