# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class aten_softplus(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.softplus(x)


class TestSoftplus(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(2, 4, 224, 224),)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_softplus(self, ie_device, precision, ir_version):
        self._test(aten_softplus(), "aten::softplus",
                   ie_device, precision, ir_version)


class TestSoftplusTail(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.concatenate((np.linspace(-100, 100, 401, dtype=np.float32),
                                np.array([-np.inf, np.inf, np.nan], dtype=np.float32))),)

    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("beta,threshold", [(1.0, 20.0), (0.5, 10.0), (2.0, 4.0)])
    def test_softplus_tail(self, ie_device, precision, ir_version, beta, threshold):
        class SoftplusTail(torch.nn.Module):
            def forward(self, value):
                return torch.nn.functional.softplus(value, beta=beta, threshold=threshold).sqrt()

        self._test(SoftplusTail(), "aten::softplus", ie_device, precision, ir_version,
                   trace_model=True, custom_eps=1e-4)


class TestSoftplusBfloat16Constants(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.linspace(-12, 12, 192, dtype=np.float32).reshape(2, 8, 12),)

    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_bfloat16_coefficients(self, ie_device, precision, ir_version):
        class SoftplusCoefficients(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("coefficients", torch.tensor([0.2041015625, -1.046875], dtype=torch.bfloat16))

            def forward(self, value):
                coefficients = torch.nn.functional.softplus(self.coefficients)
                shifted = coefficients + torch.tensor(0.5, dtype=torch.bfloat16)
                return value * coefficients[0], value * shifted[1]

        self._test(SoftplusCoefficients(), "aten::softplus", ie_device, precision, ir_version,
                   trace_model=True, custom_eps=1e-4)
