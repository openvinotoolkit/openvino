# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import openvino as ov
import pytest
import torch


pytestmark = [pytest.mark.precommit, pytest.mark.nightly]


class IndexModel(torch.nn.Module):
    __constants__ = ["pattern"]

    def __init__(self, pattern):
        super().__init__()
        self.pattern = pattern

    def forward(self, x, i, j):
        if self.pattern == "consecutive":
            return torch.ops.aten.index(x, [i, j])
        if self.pattern == "separated":
            return torch.ops.aten.index(x, [i, None, j])
        return torch.ops.aten.index(x, [None, i])


def _compare(compiled, model, inputs):
    expected = model(*inputs).numpy()
    runtime = {port: inputs[index].numpy() for index, port in enumerate(compiled.inputs)}
    np.testing.assert_array_equal(compiled(runtime)[0], expected)


@pytest.mark.parametrize("capture", ["export", "script"])
@pytest.mark.parametrize("pattern,vector_index", [
    ("consecutive", False),
    ("consecutive", True),
    ("separated", False),
    ("separated", True),
    ("single", False),
])
def test_scalar_advanced_index(capture, pattern, vector_index):
    model = IndexModel(pattern)
    x = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5)
    i = torch.tensor(1)
    j = torch.tensor([0, 2]) if vector_index else torch.tensor(2)
    inputs = (x, i, j)
    program = torch.export.export(model, inputs) if capture == "export" else torch.jit.script(model)
    converted = ov.convert_model(program, example_input=inputs if capture == "script" else None)
    expected = model(*inputs).numpy()
    assert converted.output(0).get_partial_shape().rank.get_length() == expected.ndim
    compiled = ov.Core().compile_model(converted, "CPU")
    _compare(compiled, model, inputs)


@pytest.mark.parametrize("pattern,indices", [
    pytest.param("consecutive", (1, 2), id="scalar"),
    pytest.param("consecutive", ([[0], [2]], [[0, 2]]), id="broadcast"),
    pytest.param("separated", ([[0], [2]], 2), id="mixed-separated"),
])
def test_scalar_advanced_index_dynamic(pattern, indices):
    model = IndexModel(pattern)
    x = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5)
    i, j = (torch.tensor(index) for index in indices)
    dims = {axis: torch.export.Dim("dim" + str(axis), min=3, max=9) for axis in range(3)}
    program = torch.export.export(model, (x, i, j), dynamic_shapes=(dims, None, None))
    compiled = ov.Core().compile_model(ov.convert_model(program), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    for delta, shape in enumerate(((3, 4, 5), (5, 6, 7))):
        x = torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(shape)
        _compare(compiled, model, (x, i + delta, j + delta))
