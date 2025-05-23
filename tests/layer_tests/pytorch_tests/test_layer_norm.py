# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest
import numpy as np


class TestLayerNorm(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(20, 5, 10, 10).astype(np.float32),)

    def create_model(self, normalized_shape, weight, bias, eps):
        import torch
        import torch.nn.functional as F

        if weight == "ones":
            weight = torch.ones(normalized_shape)
        
        if weight == "random":
            weight = torch.randn(normalized_shape)
        
        if bias == "zeros":
            bias = torch.zeros(normalized_shape)

        if bias == "random":
            bias = torch.randn(normalized_shape)

        class aten_layer_norm(torch.nn.Module):
            def __init__(self, normalized_shape, weight, bias, eps):
                super(aten_layer_norm, self).__init__()
                self.normalized_shape = normalized_shape
                self.weight = weight
                self.bias = bias
                self.eps = eps

            def forward(self, x):
                return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        ref_net = None

        return aten_layer_norm(normalized_shape, weight, bias, eps), ref_net, "aten::layer_norm"

    @pytest.mark.parametrize("normalized_shape", [[10,], [10, 10], [5, 10, 10]])
    @pytest.mark.parametrize("weight", [None, "ones", "random"])
    @pytest.mark.parametrize("bias", [None, "zeros", "random"])
    @pytest.mark.parametrize("eps", [1e-5, 0.005])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_layer_norm(self, normalized_shape, weight, bias, eps, ie_device, precision, ir_version):
        self._test(*self.create_model(normalized_shape, weight, bias, eps), ie_device, precision, ir_version)