# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Verify the OV_FAST_INFER binding path accepts bf16 positional inputs.

execute._torch_to_numpy hands a bf16 torch tensor to OV as a float16 numpy
array of the same bits. runtime_hooks.infer_with_pa used to wrap that array
as an f16 ov.Tensor, which set_tensor rejects on a bf16 port; the exception
fell back to plain req.infer and dropped the binding cache on every call, so
the fast path never engaged for such a graph. Results stayed correct either
way, so this checks that the input is actually bound, and still zero-copy.

No model download needed: a one-op bf16 model stands in for a PA partition.
"""

import numpy as np
import openvino as ov
import openvino.opset13 as ops
import pytest
import torch

from openvino.frontend.pytorch.torchdynamo.execute import _ov_result_to_torch, _torch_to_numpy
from openvino.frontend.pytorch.torchdynamo.vllm import runtime_hooks


@pytest.fixture
def bf16_request(monkeypatch):
    monkeypatch.setenv("OV_FAST_INFER", "1")
    runtime_hooks.clear_caches()
    param = ops.parameter([2, 4], ov.Type.bf16, name="x")
    model = ov.Model([ops.multiply(param, ops.constant(2.0, ov.Type.bf16))], [param])
    compiled = ov.Core().compile_model(model, "CPU")
    yield compiled, compiled.create_infer_request()
    runtime_hooks.clear_caches()


@pytest.mark.precommit
def test_fast_infer_binds_bf16_input(bf16_request):
    compiled, req = bf16_request
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
