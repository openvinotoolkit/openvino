# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import openvino as ov
import pytest
import torch


class FrameSliceMutation(torch.nn.Module):
    def forward(self, source):
        frames = torch.zeros_like(source)
        for index in range(source.shape[1]):
            frames[:, index] = source[:, index]
        selected = frames[0, source[0, :, 0] > 0]
        output = torch.zeros_like(source)
        output[0, :selected.size(0), :] = selected
        return output


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("trace", [False, True])
def test_slice_mutation_uses_updated_shapes_after_loop(trace):
    generator = torch.Generator().manual_seed(42)
    example = torch.randn(2, 5, 4, generator=generator)
    module = FrameSliceMutation()
    model = torch.jit.trace(torch.jit.script(module), example) if trace else torch.jit.script(module)
    converted = ov.convert_model(model, example_input=(example,))
    compiled = ov.Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    for batch, length in ((2, 5), (1, 7), (3, 4)):
        source = torch.randn(batch, length, 4, generator=generator)
        expected = model(source).numpy()
        actual = next(iter(compiled([source.numpy()]).values()))
        np.testing.assert_array_equal(actual, expected)
