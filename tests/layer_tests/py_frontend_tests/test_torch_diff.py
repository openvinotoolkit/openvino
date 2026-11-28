# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import openvino as ov
import pytest
import torch


class Diff(torch.nn.Module):
    __constants__ = ["order", "axis", "prepend", "append"]

    def __init__(self, order, axis, prepend, append):
        super().__init__()
        self.order = order
        self.axis = axis
        self.prepend = prepend
        self.append = append

    def forward(self, x, before, after):
        return torch.diff(x, n=self.order, dim=self.axis,
                          prepend=before if self.prepend else None,
                          append=after if self.append else None)


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("capture", ["export", "script"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.uint8, torch.bool])
@pytest.mark.parametrize("order,axis,prepend,append", [
    (0, -1, True, True),
    (1, -1, False, False),
    (2, 0, False, False),
    (1, 1, True, False),
    (2, -1, True, True),
    (5, -1, False, False),
])
def test_torch_diff(capture, dtype, order, axis, prepend, append, monkeypatch):
    model = Diff(order, axis, prepend, append).eval()
    x = torch.arange(12).reshape(3, 4).to(dtype)
    if dtype == torch.uint8:
        x = torch.tensor([[255, 0, 1, 128], [2, 250, 3, 0], [240, 4, 128, 1]], dtype=dtype)
    shape = list(x.shape)
    shape[axis] = 2
    before = torch.ones(shape, dtype=dtype)
    after = torch.zeros(shape, dtype=dtype)
    inputs = (x, before, after)
    if capture == "export":
        program = torch.export.export(model, inputs)
        monkeypatch.setattr(torch.export.ExportedProgram, "run_decompositions",
                            lambda *args, **kwargs: pytest.fail("Conversion must not run decompositions"))
    else:
        program = torch.jit.script(model)
    converted = ov.convert_model(program, example_input=inputs if capture == "script" else None)
    compiled = ov.Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    runtime = {port: inputs[i].numpy() for i, port in enumerate(compiled.inputs)}
    actual = compiled(runtime)[0]
    expected = model(*inputs).numpy()
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.precommit
@pytest.mark.nightly
def test_torch_diff_dynamic_length(monkeypatch):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.diff(x, n=2, dim=-1)

    program = torch.export.export(Model(), (torch.randn(2, 4),),
                                  dynamic_shapes=({1: torch.export.Dim("length", min=4, max=16)},))
    monkeypatch.setattr(torch.export.ExportedProgram, "run_decompositions",
                        lambda *args, **kwargs: pytest.fail("Conversion must not run decompositions"))
    converted = ov.convert_model(program)
    converted.reshape([2, -1])
    compiled = ov.Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    for length in [0, 1, 2, 3, 7]:
        x = torch.arange(2 * length, dtype=torch.float32).reshape(2, length)
        np.testing.assert_array_equal(compiled(x.numpy())[0], Model()(x).numpy())


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("capture", ["export", "script"])
@pytest.mark.parametrize("prepend_dtype,input_dtype,prepend_value,input_value", [
    (torch.int32, torch.float16, 2049, 2048),
    (torch.float16, torch.int32, 2048, 2049),
])
def test_torch_diff_promotes_all_boundaries(capture, prepend_dtype, input_dtype, prepend_value, input_value):
    model = Diff(1, -1, True, True).eval()
    inputs = (torch.tensor([input_value], dtype=input_dtype),
              torch.tensor([prepend_value], dtype=prepend_dtype),
              torch.tensor([2048], dtype=torch.float32))
    program = torch.export.export(model, inputs) if capture == "export" else torch.jit.script(model)
    converted = ov.convert_model(program, example_input=inputs if capture == "script" else None)
    expected = model(*inputs).numpy()
    reference = ov.Tensor(np.empty_like(expected))
    assert converted.evaluate([reference], [ov.Tensor(value.numpy()) for value in inputs])
    np.testing.assert_array_equal(reference.data, expected)
    compiled = ov.Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    actual = compiled([x.numpy() for x in inputs])[0]
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("capture", ["export", "script"])
@pytest.mark.parametrize("order,input_dtype,out_dtype", [
    (0, torch.float32, torch.float32),
    (1, torch.float32, torch.float32),
    (1, torch.int64, torch.int8),
    (1, torch.int64, torch.float32),
])
def test_torch_diff_out(capture, order, input_dtype, out_dtype):
    class Model(torch.nn.Module):
        __constants__ = ["order"]

        def __init__(self):
            super().__init__()
            self.order = order

        def forward(self, x, out):
            value = torch.diff(x, n=self.order, out=out)
            return value, out * 2

    values = [0, 200, 500] if out_dtype == torch.int8 else [0, 16777217, 33554434]
    inputs = (torch.tensor(values, dtype=input_dtype), torch.zeros(3 - order, dtype=out_dtype))
    model = Model()
    program = torch.export.export(model, inputs) if capture == "export" else torch.jit.script(model)
    converted = ov.convert_model(program, example_input=inputs if capture == "script" else None)
    compiled = ov.Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    actual = list(compiled([inputs[i].numpy() for i in range(len(compiled.inputs))]).values())
    expected = model(*(value.clone() for value in inputs))
    assert len(actual) == len(expected)
    for result, reference in zip(actual, expected):
        assert result.dtype == reference.numpy().dtype
        np.testing.assert_array_equal(result, reference.numpy())


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("capture", ["export", "script"])
@pytest.mark.parametrize("order,axis", [(1, -1), (2, 0)])
def test_torch_diff_polar(capture, order, axis):
    class Model(torch.nn.Module):
        __constants__ = ["order", "axis"]

        def __init__(self):
            super().__init__()
            self.order = order
            self.axis = axis

        def forward(self, magnitude, angle):
            value = torch.polar(magnitude, angle)
            boundary = torch.polar(magnitude.narrow(self.axis, 0, 1), angle.narrow(self.axis, 0, 1))
            return torch.view_as_real(torch.diff(value, n=self.order, dim=self.axis,
                                                 prepend=boundary))

    inputs = (torch.arange(1, 9, dtype=torch.float32).reshape(2, 4),
              torch.arange(-4, 4, dtype=torch.float32).reshape(2, 4) / 3)
    model = Model()
    program = torch.export.export(model, inputs) if capture == "export" else torch.jit.script(model)
    converted = ov.convert_model(program, example_input=inputs if capture == "script" else None)
    compiled = ov.Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    np.testing.assert_allclose(compiled([x.numpy() for x in inputs])[0], model(*inputs).numpy(),
                               rtol=1e-5, atol=1e-5)


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("prepend", [True, False])
@pytest.mark.parametrize("input_dtype,boundary_dtype", [
    (torch.bool, torch.bool),
    (torch.bool, torch.int32),
    (torch.bool, torch.uint8),
    (torch.uint8, torch.uint8),
    (torch.uint8, torch.int32),
    (torch.uint8, torch.bool),
])
def test_torch_diff_script_branch_promotion(prepend, input_dtype, boundary_dtype):
    class Model(torch.nn.Module):
        __constants__ = ["prepend"]

        def __init__(self):
            super().__init__()
            self.prepend = prepend

        def forward(self, x, boundary, take_branch):
            if take_branch:
                return torch.diff(x, n=2, prepend=boundary if self.prepend else None,
                                  append=boundary if not self.prepend else None)
            return torch.diff(x, prepend=boundary if self.prepend else None,
                              append=boundary if not self.prepend else None)

    x = torch.tensor([255, 0, 128, 1], dtype=torch.uint8).to(input_dtype)
    boundary = torch.tensor([3, 250], dtype=torch.uint8).to(boundary_dtype)
    model = Model()
    inputs = (x, boundary, torch.tensor(True))
    converted = ov.convert_model(torch.jit.script(model), example_input=inputs)
    compiled = ov.Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    for take_branch in [True, False]:
        inputs = (x, boundary, torch.tensor(take_branch))
        actual = compiled([value.numpy() for value in inputs])[0]
        expected = model(*inputs).numpy()
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
