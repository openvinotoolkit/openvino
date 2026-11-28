# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
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


class aten_softplus_runtime_beta(torch.nn.Module):
    def forward(self, x, beta):
        return torch.nn.functional.softplus(x, beta=beta.item(), threshold=1.5)


class TestSoftplusRuntimeBeta(PytorchLayerTest):
    def _prepare_input(self, beta):
        return (self.random.randn(2, 3, 16, 16) * 4, np.array(beta, dtype=np.float32))

    @pytest.mark.parametrize("beta", [1.0, 2.0])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_softplus_runtime_beta(self, beta, ie_device, precision, ir_version):
        self._test(aten_softplus_runtime_beta(), "aten::softplus",
                   ie_device, precision, ir_version, kwargs_to_prepare_input={"beta": beta})


class aten_softplus_fp16(torch.nn.Module):
    def __init__(self, beta, threshold):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x):
        return torch.nn.functional.softplus(x.to(torch.float16), beta=self.beta, threshold=self.threshold)


class TestSoftplusFP16(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array([0.0, 1.0, -1.0, 0.5, 2.0, -3.0], dtype=np.float32),)

    @pytest.mark.parametrize("beta,threshold", [
        (1.0, 0.9999),
        (65536.0, 20.0),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_softplus_fp16(self, beta, threshold, ie_device, precision, ir_version):
        self._test(aten_softplus_fp16(beta, threshold), "aten::softplus",
                   ie_device, precision, ir_version)


class TestSoftplusFP16Input(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array([0.0, 1.0, -1.0, 0.5, 2.0, -3.0], dtype=np.float16),)

    @pytest.mark.parametrize("beta,threshold", [
        (1.0, 0.9999),
        (65536.0, 20.0),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_softplus_fp16_input(self, beta, threshold, ie_device, precision, ir_version):
        self._test(aten_softplus_beta_threshold(beta, threshold), "aten::softplus",
                   ie_device, precision, ir_version)
