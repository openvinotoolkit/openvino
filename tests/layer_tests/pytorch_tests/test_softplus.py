# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class aten_softplus(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.softplus(x)


class aten_softplus_beta_threshold(torch.nn.Module):
    def __init__(self, beta, threshold):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x):
        return torch.nn.functional.softplus(x, beta=self.beta, threshold=self.threshold)


class TestSoftplus(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(2, 4, 224, 224),)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_softplus(self, ie_device, precision, ir_version):
        self._test(aten_softplus(), "aten::softplus",
                   ie_device, precision, ir_version)


class TestSoftplusBetaThreshold(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(2, 3, 16, 16) * 4,)

    @pytest.mark.parametrize("beta,threshold", [
        (2.0, 20.0),
        (0.5, 20.0),
        (1.0, 1.0),
        (2.0, 1.5),
        (3, 2),
        (-1.0, 20.0),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_softplus_beta_threshold(self, beta, threshold, ie_device, precision, ir_version):
        self._test(aten_softplus_beta_threshold(beta, threshold), "aten::softplus",
                   ie_device, precision, ir_version)
