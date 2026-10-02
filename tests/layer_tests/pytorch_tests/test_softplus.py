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
