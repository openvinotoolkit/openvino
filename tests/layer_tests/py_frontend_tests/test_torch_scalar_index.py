# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import openvino as ov
import pytest
import torch


pytestmark = [pytest.mark.precommit, pytest.mark.nightly]


@pytest.mark.parametrize("capture", ["export", "script"])
@pytest.mark.parametrize("pattern", ["consecutive", "separated", "single"])
@pytest.mark.parametrize("vector_index", [False, True])
def test_scalar_advanced_index(capture, pattern, vector_index):
    class Model(torch.nn.Module):
        __constants__ = ["pattern"]

        def __init__(self):
            super().__init__()
            self.pattern = pattern

        def forward(self, x, i, j):
            if self.pattern == "consecutive":
                return torch.ops.aten.index(x, [i, j])
            if self.pattern == "separated":
                return torch.ops.aten.index(x, [i, None, j])
            return torch.ops.aten.index(x, [None, i])

    model = Model()
    x = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5)
    i = torch.tensor(1)
    j = torch.tensor([0, 2]) if vector_index else torch.tensor(2)
    inputs = (x, i, j)
    program = torch.export.export(model, inputs) if capture == "export" else torch.jit.script(model)
    converted = ov.convert_model(program, example_input=inputs if capture == "script" else None)
    expected = model(*inputs).numpy()
    assert converted.output(0).get_partial_shape().rank.get_length() == expected.ndim
    compiled = ov.Core().compile_model(converted, "CPU")
    runtime = {port: inputs[index].numpy() for index, port in enumerate(compiled.inputs)}
    np.testing.assert_array_equal(compiled(runtime)[0], expected)
