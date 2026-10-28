# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import openvino as ov
import pytest
import torch


pytestmark = [pytest.mark.precommit, pytest.mark.nightly]


@pytest.mark.parametrize("capture", ["export", "script"])
@pytest.mark.parametrize("dtype", [torch.bool, torch.int32, torch.int64, torch.float32])
@pytest.mark.parametrize("explicit_dtype", [False, True])
def test_cumsum_dtype(capture, dtype, explicit_dtype, monkeypatch):
    class Model(torch.nn.Module):
        __constants__ = ["explicit_dtype"]

        def __init__(self):
            super().__init__()
            self.explicit_dtype = explicit_dtype

        def forward(self, x):
            if self.explicit_dtype:
                return torch.cumsum(x, -1, dtype=torch.float32)
            return torch.cumsum(x, -1)

    model = Model().eval()
    x = torch.tensor([[0, 1, 1, 0, 1], [1, 0, 1, 1, 1]], dtype=dtype)
    program = torch.export.export(model, (x,)) if capture == "export" else torch.jit.script(model)
    monkeypatch.setattr(torch.export.ExportedProgram, "run_decompositions",
                        lambda *args, **kwargs: pytest.fail("Conversion must not run decompositions"))
    converted = ov.convert_model(program, example_input=(x,) if capture == "script" else None)
    expected = model(x).numpy()
    assert converted.output(0).get_element_type().to_dtype() == expected.dtype
    compiled = ov.Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    np.testing.assert_array_equal(compiled(x.numpy())[0], expected)


@pytest.mark.parametrize("dtype", [torch.bool, torch.int32])
def test_cumsum_out_dtype(dtype):
    class Model(torch.nn.Module):
        def forward(self, x, out):
            return torch.cumsum(x, -1, out=out)

    x = torch.tensor([0, 1, 1, 0, 1], dtype=dtype)
    out = torch.zeros(5, dtype=torch.int32)
    model = Model()
    converted = ov.convert_model(torch.jit.script(model), example_input=(x, out))
    assert converted.output(0).get_element_type() == ov.Type.i32
    compiled = ov.Core().compile_model(converted, "CPU")
    inputs = {port: tensor.numpy() for port, tensor in zip(compiled.inputs, (x, out))}
    np.testing.assert_array_equal(compiled(inputs)[0], model(x, out).numpy())
