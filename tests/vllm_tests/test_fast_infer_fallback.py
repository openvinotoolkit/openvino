# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Verify the OV_FAST_INFER path runs inference at most once per call.

runtime_hooks.infer_with_pa falls back to plain req.infer when binding the
inputs fails. Once req.infer() has run, the stateful ops (GDN convolution
and recurrent state, the PA KV cache) have already been updated in place, so
re-running on any later error would advance them twice for one step. A
failure after inference must propagate instead.

No model download needed: a one-op model stands in for a PA partition, and
a proxy request injects the failures.
"""

import openvino as ov
import openvino.opset13 as ops
import pytest

from openvino.frontend.pytorch.torchdynamo.vllm import runtime_hooks


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


@pytest.fixture
def compiled_model(monkeypatch):
    monkeypatch.setenv("OV_FAST_INFER", "1")
    runtime_hooks.clear_caches()
    param = ops.parameter([2, 4], ov.Type.f32, name="x")
    model = ov.Model([ops.multiply(param, ops.constant(2.0, ov.Type.f32))], [param])
    yield ov.Core().compile_model(model, "CPU")
    runtime_hooks.clear_caches()


def _inputs(compiled):
    return {compiled.inputs[0]: ov.Tensor(ov.Type.f32, [2, 4])}


@pytest.mark.precommit
def test_output_failure_does_not_rerun_inference(compiled_model):
    req = _Request(compiled_model.create_infer_request(), fail_output=True)
    with pytest.raises(RuntimeError, match="injected output failure"):
        runtime_hooks.infer_with_pa(req, compiled_model, _inputs(compiled_model))
    assert req.runs == 1, f"inference ran {req.runs} times for one call"


@pytest.mark.precommit
def test_binding_failure_falls_back_once(compiled_model):
    req = _Request(compiled_model.create_infer_request(), fail_bind=True)
    runtime_hooks.infer_with_pa(req, compiled_model, _inputs(compiled_model))
    assert req.runs == 1, f"inference ran {req.runs} times for one call"
    assert id(req) not in runtime_hooks._fastinfer_bound_ids
