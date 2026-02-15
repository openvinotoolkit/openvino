# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestMatMul(PytorchLayerTest):
    def _prepare_input(self, m1_shape=(2, 2), m2_shape=(2, 2), bias_shape=None):
        import numpy as np
        if bias_shape is None:
            return (np.random.randn(*m1_shape).astype(np.float32), np.random.randn(*m2_shape).astype(np.float32))
        else:
            return (np.random.randn(*m1_shape).astype(np.float32), np.random.randn(*m2_shape).astype(np.float32), np.random.randn(*bias_shape).astype(np.float32))

    def create_model(self, is_bias):
        import torch

        class aten_mm(torch.nn.Module):
            def __init__(self, is_bias):
                super(aten_mm, self).__init__()
                self.forward = self.forward2 if is_bias else self.forward1

            def forward1(self, m1, m2):
                return torch.nn.functional.linear(m1, m2)

            def forward2(self, m1, m2, bias):
                return torch.nn.functional.linear(m1, m2, bias)

        ref_net = None

        return aten_mm(is_bias), ref_net, "aten::linear"

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
        import numpy as np
        return (np.random.randn(1, 15, 10).astype(np.float32), np.random.randn(66, 10).astype(np.float32))

    def create_model(self):
        import torch

        class aten_mm(torch.nn.Module):
            def __init__(self):
                super(aten_mm, self).__init__()
                self.bias = [torch.randn(22),
                             torch.randn(22),
                             torch.randn(22)]

            def forward(self, m1, m2):
                m2 = m2.reshape([66, -1])
                return torch.nn.functional.linear(m1, m2, torch.cat(self.bias, 0))

        return aten_mm(), None, "aten::linear"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_linear_bias_list(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   trace_model=True, freeze_model=False)
