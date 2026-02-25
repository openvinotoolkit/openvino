# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestPRelu(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(1, 3, 224, 224),)

    def create_model(self, alpha):
        import torch
        import torch.nn.functional as F

        class aten_prelu(torch.nn.Module):
            def __init__(self, alpha):
                super().__init__()
                self.alpha = torch.Tensor([alpha])

            def forward(self, x):
                return x, F.prelu(x, self.alpha)


        return aten_prelu(alpha), "aten::prelu"

    @pytest.mark.parametrize("alpha", [0.01, 1.01, -0.01])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_prelu(self, alpha, ie_device, precision, ir_version):
        self._test(*self.create_model(alpha), ie_device, precision, ir_version)
