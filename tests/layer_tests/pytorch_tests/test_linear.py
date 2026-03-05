# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestMatMul(PytorchLayerTest):
    def _prepare_input(self, m1_shape=(2, 2), m2_shape=(2, 2), bias_shape=None):
        if bias_shape is None:
            return (self.random.randn(*m1_shape), self.random.randn(*m2_shape))
        else:
            return (self.random.randn(*m1_shape), self.random.randn(*m2_shape), self.random.randn(*bias_shape))

    def create_model(self, is_bias):
        import torch

        class aten_mm(torch.nn.Module):
            def __init__(self, is_bias):
                super().__init__()
                self.forward = self.forward2 if is_bias else self.forward1

            def forward1(self, m1, m2):
                return torch.nn.functional.linear(m1, m2)

            def forward2(self, m1, m2, bias):
                return torch.nn.functional.linear(m1, m2, bias)


        return aten_mm(is_bias), "aten::linear"

    @pytest.mark.parametrize("kwargs_to_prepare_input", [
        {'m1_shape': [9], 'm2_shape': [10, 9]},
        {'m1_shape': [9], 'm2_shape': [9]},
        {'m1_shape': [3, 9], 'm2_shape': [10, 9]},
        {'m1_shape': [3, 9], 'm2_shape': [9]},
        {'m1_shape': [2, 3, 9], 'm2_shape': [10, 9]},
        {'m1_shape': [2, 3, 9], 'm2_shape': [9]},
        {'m1_shape': [9], 'm2_shape': [10, 9], 'bias_shape': [10]},
        {'m1_shape': [3, 9], 'm2_shape': [10, 9], 'bias_shape': [10]},
        {'m1_shape': [2, 3, 9], 'm2_shape': [10, 9], 'bias_shape': [10]},
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_matmul(self, kwargs_to_prepare_input, ie_device, precision, ir_version):
        self._test(*self.create_model(len(kwargs_to_prepare_input) == 3), ie_device, precision, ir_version,
                   kwargs_to_prepare_input=kwargs_to_prepare_input)


class TestLinearBiasList(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(1, 15, 10), self.random.randn(66, 10))

    def create_model(self):
        import torch

        class aten_mm(torch.nn.Module):
            def __init__(self, rng):
                super().__init__()
                self.bias = [rng.torch_randn(22),
                             rng.torch_randn(22),
                             rng.torch_randn(22)]

            def forward(self, m1, m2):
                m2 = m2.reshape([66, -1])
                return torch.nn.functional.linear(m1, m2, torch.cat(self.bias, 0))

        return aten_mm(self.random), "aten::linear"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_linear_bias_list(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   trace_model=True, freeze_model=False)
