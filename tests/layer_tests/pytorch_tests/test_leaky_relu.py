# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest, skip_check


class TestLeakyRelu(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(1, 3, 224, 224),)

    def create_model(self, alpha, inplace):
        import torch
        import torch.nn.functional as F

        class aten_leaky_relu(torch.nn.Module):
            def __init__(self, alpha, inplace):
                super().__init__()
                self.alpha = alpha
                self.inplace = inplace

            def forward(self, x):
                return x, F.leaky_relu(x, self.alpha, inplace=self.inplace)


        return aten_leaky_relu(alpha, inplace), "aten::leaky_relu" if not inplace else "aten::leaky_relu_"

    @pytest.mark.parametrize("alpha", [0.01, 1.01, -0.01])
    @pytest.mark.parametrize("inplace", [skip_check(True), False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_leaky_relu(self, alpha, inplace, ie_device, precision, ir_version):
        self._test(*self.create_model(alpha, inplace), ie_device, precision, ir_version)
