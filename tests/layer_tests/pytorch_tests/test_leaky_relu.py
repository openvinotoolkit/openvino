# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest, skip_if_export


class TestLeakyRelu(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    def create_model(self, alpha, inplace):
        import torch
        import torch.nn.functional as F

        class aten_leaky_relu(torch.nn.Module):
            def __init__(self, alpha, inplace):
                super(aten_leaky_relu, self).__init__()
                self.alpha = alpha
                self.inplace = inplace

            def forward(self, x):
                return x, F.leaky_relu(x, self.alpha, inplace=self.inplace)

        ref_net = None

        return aten_leaky_relu(alpha, inplace), ref_net, "aten::leaky_relu" if not inplace else "aten::leaky_relu_"

    @pytest.mark.parametrize("alpha", [0.01, 1.01, -0.01])
    @pytest.mark.parametrize("inplace", [skip_if_export(True), False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_leaky_relu(self, alpha, inplace, ie_device, precision, ir_version):
        self._test(*self.create_model(alpha, inplace), ie_device, precision, ir_version)
