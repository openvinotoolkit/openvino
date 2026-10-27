# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import numpy as np
import openvino as ov
import pytest
import torch
from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("scope", ["branch", "nested", "loop"])
def test_torchscript_subgraph_convolution_constants(dimension, groups, scope):
    class Decoder(torch.nn.Module):
        __constants__ = ["scope"]

        def __init__(self):
            super().__init__()
            conv = getattr(torch.nn, f"Conv{dimension}d")
            self.conv = conv(4, 4, 2, stride=2, padding=1, dilation=2, groups=groups)
            self.context_size = 2
            self.scope = scope
            self.repeat = 2

        def forward(self, x: torch.Tensor):
            if self.context_size > 1:
                if self.scope == "loop":
                    for _ in range(self.repeat):
                        x = self.conv(x)
                else:
                    x = self.conv(x)
            return x

    class Wrapper(torch.nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder
            self.enabled = True

        def forward(self, x: torch.Tensor):
            if self.enabled:
                return self.decoder(x)
            return x

    torch.manual_seed(0)
    model = Wrapper(Decoder())
    if scope == "nested":
        model = Wrapper(model)
    model.eval()
    source = torch.jit.script(model)
    inputs = torch.randn((1, 4) + (6,) * dimension)
    converted = ov.convert_model(source, example_input=inputs)
    compiled = ov.Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    actual = compiled(inputs.numpy())[0]
    expected = source(inputs).detach().numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("scope", ["branch", "nested", "loop"])
def test_torchscript_subgraph_computed_convolution_constants(scope):
    class Model(torch.nn.Module):
        __constants__ = ["scope"]

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 4, 3, 3))
            self.kernel_size = 3
            self.enabled = True
            self.inner_enabled = True
            self.repeat = 2
            self.scope = scope

        def convolve(self, x: torch.Tensor, stride: int, padding: int):
            return torch.nn.functional.conv2d(x, self.weight, stride=[stride, stride], padding=[padding, padding])

        def forward(self, x: torch.Tensor):
            stride = self.kernel_size - 1
            padding = self.kernel_size // 2
            if self.enabled:
                if self.scope == "loop":
                    for _ in range(self.repeat):
                        x = self.convolve(x, stride, padding)
                elif self.scope == "nested":
                    if self.inner_enabled:
                        x = self.convolve(x, stride, padding)
                else:
                    x = self.convolve(x, stride, padding)
            return x

    torch.manual_seed(0)
    source = torch.jit.script(Model().eval())
    inputs = torch.randn(1, 4, 9, 9)
    converted = ov.convert_model(source, example_input=inputs)
    compiled = ov.Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    actual = compiled(inputs.numpy())[0]
    expected = source(inputs).detach().numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("outer,inner", [(False, False), (False, True), (True, False), (True, True)])
def test_torchscript_nested_constant_mutation_before_append(outer, inner):
    class Model(torch.nn.Module):
        def forward(self, x: torch.Tensor, weight: torch.Tensor, outer: torch.Tensor, inner: torch.Tensor):
            if bool(outer):
                weight.add_(3)
                if bool(inner):
                    values = [x]
                    values.append(weight)
                    return torch.cat(values)
                return weight
            return x

    inputs = (torch.tensor([10., 20.]), torch.tensor([1., 2.]), torch.tensor(outer), torch.tensor(inner))
    source = torch.jit.script(Model().eval())
    expected = source(*deepcopy(inputs)).numpy()
    frontend = ov.frontend.FrontEndManager().load_by_framework("pytorch")
    decoder = TorchScriptPythonDecoder(source)
    input_model = frontend.load(decoder)
    places = input_model.get_inputs()
    for place, value in zip(places, inputs):
        input_model.set_partial_shape(place, ov.PartialShape(value.shape))
        input_model.set_element_type(place, ov.Type.boolean if value.dtype == torch.bool else ov.Type.f32)
    input_model.set_tensor_value(places[1], inputs[1].numpy())
    converted = frontend.convert(input_model)
    compiled = ov.Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    actual = compiled([value.numpy() for value in (inputs[0], inputs[2], inputs[3])])[0]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("container", ["list", "tuple"])
@pytest.mark.parametrize("positive", [False, True])
def test_torchscript_nested_runtime_sequence_values(container, positive):
    class ListModel(torch.nn.Module):
        def forward(self, x: torch.Tensor):
            values = [x, x + 1]
            if x.sum() > 0:
                x.add_(2)
            else:
                x.sub_(2)
            return values[0] + values[1]

    class TupleModel(torch.nn.Module):
        def forward(self, x: torch.Tensor):
            values = (x, x + 1)
            if x.sum() > 0:
                x.add_(2)
            else:
                x.sub_(2)
            return values[0] + values[1]

    source = torch.jit.script((ListModel() if container == "list" else TupleModel()).eval())
    inputs = torch.full((2, 3), 1.0 if positive else -1.0)
    expected = source(inputs.clone()).numpy()
    converted = ov.convert_model(source, example_input=inputs.clone())
    compiled = ov.Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    actual = compiled(inputs.numpy())[0]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("iterations", [1, 2, 3])
def test_torchscript_loop_carried_constant_sequence(iterations):
    class Model(torch.nn.Module):
        __constants__ = ["iterations"]

        def __init__(self):
            super().__init__()
            self.iterations = iterations

        def forward(self, x: torch.Tensor, weight: torch.Tensor):
            out = x.clone()
            for _ in range(self.iterations):
                values = [weight]
                out = out + torch.cat(values)
                weight.add_(3)
            return out

    inputs = (torch.tensor([10., 20.]), torch.tensor([1., 2.]))
    source = torch.jit.script(Model().eval())
    expected = source(*deepcopy(inputs)).numpy()
    frontend = ov.frontend.FrontEndManager().load_by_framework("pytorch")
    decoder = TorchScriptPythonDecoder(source)
    input_model = frontend.load(decoder)
    places = input_model.get_inputs()
    for place, value in zip(places, inputs):
        input_model.set_partial_shape(place, ov.PartialShape(value.shape))
        input_model.set_element_type(place, ov.Type.f32)
    input_model.set_tensor_value(places[1], inputs[1].numpy())
    converted = frontend.convert(input_model)
    compiled = ov.Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    actual = compiled(inputs[0].numpy())[0]
    np.testing.assert_array_equal(actual, expected)
