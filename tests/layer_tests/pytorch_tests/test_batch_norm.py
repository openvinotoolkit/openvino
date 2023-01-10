# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytorch_layer_test_class import PytorchLayerTest


class TestBatchNorm(PytorchLayerTest):
    def _prepare_input(self, ndim=4):
        import numpy as np
        shape5d = [20, 6, 10, 10, 10]
        shape = shape5d[:ndim]
        return (np.random.randn(*shape).astype(np.float32),)

    def create_model(self, weights, bias, eps):

        import torch
        import torch.nn.functional as F

        class aten_batch_norm_inference(torch.nn.Module):
            def __init__(self, weights=True, bias=True, eps=1e-05):
                super(aten_batch_norm_inference, self).__init__()
                self.weight = torch.randn(6) if weights else None
                self.bias = torch.randn(6) if bias else None
                self.running_mean = torch.randn(6)
                self.running_var = torch.randn(6)
                self.eps = eps

            def forward(self, x):
                return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, eps=self.eps, training=False)

        ref_net = None

        return aten_batch_norm_inference(weights, bias, eps), ref_net, "aten::batch_norm"

    @pytest.mark.parametrize("weights", [True, False])
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("eps", [1.0, 0.00005, 0.5, 0.042])
    @pytest.mark.parametrize("kwargs_to_prepare_input", [
        {"ndim": 3},
        {'ndim': 4},
        {"ndim": 5}
    ])
    @pytest.mark.nightly
    def test_batch_norm(self, weights, bias, eps, ie_device, precision, ir_version, kwargs_to_prepare_input):
        self._test(*self.create_model(weights, bias, eps),
                   ie_device, precision, ir_version, kwargs_to_prepare_input=kwargs_to_prepare_input, dynamic_shapes=False)