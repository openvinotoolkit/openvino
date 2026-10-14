# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import numpy as np
import openvino as ov
import pytest
import torch


class ContainerMutation(torch.nn.Module):
    def __init__(self, nested, mutation):
        super().__init__()
        self.nested = nested
        self.mutation = mutation

    def forward(self, source, cache):
        left, right = cache[0] if self.nested else cache
        before = left + right
        left_view = left[:, :, :2]
        right_view = right[:, :, :2]
        if self.mutation == "view":
            left_view.copy_(source[:, :, :2])
        elif self.mutation == "metadata":
            left.transpose_(1, 2)
            left.copy_(source.transpose(1, 2))
        elif self.mutation == "right":
            right.copy_(source)
        else:
            left.copy_(source)
            if self.mutation == "repeated":
                left.add_(2)
            elif self.mutation == "siblings":
                right.copy_(source + 5)
        values = cache[0] if self.nested else cache
        return before, left, right, left_view, right_view, values[0], values[1]


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("mutation", ["copy", "right", "view", "repeated", "siblings", "metadata"])
def test_torch_container_input_mutation(container, nested, mutation):
    generator = torch.Generator().manual_seed(42)

    def make_inputs(batch):
        source = torch.randn(batch, 4, 3, generator=generator)
        elements = container(torch.randn(batch, 4, 3, generator=generator) for _ in range(2))
        cache = container([elements]) if nested else elements
        return source, cache

    inputs = make_inputs(1)
    model = torch.jit.trace(ContainerMutation(nested, mutation), deepcopy(inputs),
                           strict=False, check_trace=False)
    converted = ov.convert_model(model, example_input=deepcopy(inputs))
    assert len(converted.inputs) == 3
    compiled = ov.Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    for batch in (1, 2):
        source, cache = make_inputs(batch)
        expected = model(*deepcopy((source, cache)))
        elements = cache[0] if nested else cache
        actual = list(compiled([source.numpy()] + [element.numpy() for element in elements]).values())
        assert len(actual) == len(expected)
        for reference, result in zip(expected, actual):
            np.testing.assert_allclose(result, reference.numpy(), rtol=1e-5, atol=1e-6)


class ChunkCacheMutation(torch.nn.Module):
    def forward(self, source, cache):
        b, c, x = source.chunk(3, dim=1)
        cache[0].copy_(b * x)
        return c * cache[0], cache


class SequenceViewMutation(torch.nn.Module):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    def forward(self, source, data):
        base = data.clone()
        if self.operation == "split":
            parts = torch.split_with_sizes(base, [2, 2], dim=1)
        else:
            parts = torch.unbind(base.reshape(base.shape[0], 2, 2, 3), dim=1)
        before = parts[0] + parts[1]
        parts[0].copy_(source)
        parts[1].add_(2)
        return base, before, parts[0], parts[1]


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("operation,capture", [("split", "torchscript"),
                                               ("split", "export"), ("unbind", "export")])
def test_torch_sequence_view_mutation(operation, capture):
    generator = torch.Generator().manual_seed(42)
    inputs = (torch.randn(2, 2, 3, generator=generator), torch.randn(2, 4, 3, generator=generator))
    model = SequenceViewMutation(operation)
    if capture == "torchscript":
        model = torch.jit.trace(model, deepcopy(inputs))
        converted = ov.convert_model(model, example_input=deepcopy(inputs))
    else:
        batch = torch.export.Dim("batch", min=1)
        exported = torch.export.export(model, deepcopy(inputs), dynamic_shapes=({0: batch}, {0: batch}))
        converted = ov.convert_model(exported)
    compiled = ov.Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    for batch in (1, 2):
        inputs = (torch.randn(batch, 2, 3, generator=generator), torch.randn(batch, 4, 3, generator=generator))
        expected = model(*deepcopy(inputs))
        actual = list(compiled([element.numpy() for element in inputs]).values())
        assert len(actual) == len(expected)
        for reference, result in zip(expected, actual):
            np.testing.assert_allclose(result, reference.numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.precommit
@pytest.mark.nightly
def test_torch_chunk_cache_mutation():
    generator = torch.Generator().manual_seed(42)
    source = torch.randn(1, 12, 3, generator=generator)
    cache = [torch.randn(1, 4, 3, generator=generator) for _ in range(2)]
    inputs = source, cache
    model = torch.jit.trace(ChunkCacheMutation(), deepcopy(inputs), strict=False, check_trace=False)
    converted = ov.convert_model(model, example_input=deepcopy(inputs))
    compiled = ov.Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    for batch in (1, 2):
        source = torch.randn(batch, 12, 3, generator=generator)
        cache = [torch.randn(batch, 4, 3, generator=generator) for _ in range(2)]
        output, updated_cache = model(*deepcopy((source, cache)))
        expected = [output] + updated_cache
        actual = list(compiled([source.numpy()] + [element.numpy() for element in cache]).values())
        assert len(actual) == len(expected)
        for reference, result in zip(expected, actual):
            np.testing.assert_allclose(result, reference.numpy(), rtol=1e-5, atol=1e-6)
