# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import torch
from pytorch_layer_test_class import PytorchLayerTest


class TestQuantile(PytorchLayerTest):
    def _prepare_input(self):
        input_tensor = np.random.randn(1, 3, 224, 224).astype(np.float32)
        quantile = np.array(0.5, dtype=np.float32)
        return (input_tensor, quantile)

    def create_model(self, dim=None, keepdim=False):
        class aten_quantile(torch.nn.Module):
            def __init__(self, dim, keepdim):
                super(aten_quantile, self).__init__()
                self.dim = dim
                self.keepdim = keepdim

            def forward(self, x, q):
                return torch.quantile(x, q, dim=self.dim, keepdim=self.keepdim)

        ref_net = None

        return aten_quantile(dim, keepdim), ref_net, "aten::quantile"

    @pytest.mark.parametrize("dim", [None, 0, 1, 2, 3, -1, -2, -3])
    @pytest.mark.parametrize("keepdim", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_quantile(self, dim, keepdim, ie_device, precision, ir_version):
        self._test(*self.create_model(dim, keepdim), ie_device, precision, ir_version)

