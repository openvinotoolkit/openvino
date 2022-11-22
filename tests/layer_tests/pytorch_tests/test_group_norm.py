# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytorch_layer_test_class import PytorchLayerTest


class TestGroupNorm(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(20, 6, 10, 10).astype(np.float32),)

    def create_model(self, n_groups, weights_shape=None, bias=False, eps=1e-05):

        import torch
        import torch.nn.functional as F

        class aten_group_norm(torch.nn.Module):
            def __init__(self, n_groups, weights_shape=None, bias=True, eps=1e-05):
                super(aten_group_norm, self).__init__()
                self.weight = torch.randn(weights_shape) if weights_shape else None
                self.bias = None
                if bias:
                    self.bias = torch.randn(weights_shape)
                self.n_groups = n_groups
                self.eps = eps

            def forward(self, x):
                return F.group_norm(x, self.n_groups, self.weight, self.bias, self.eps)

        ref_net = None

        return aten_group_norm(n_groups, weights_shape, bias, eps), ref_net, "aten::group_norm"

    @pytest.mark.parametrize("params",
                             [
                              {"n_groups": 3},
                              {"n_groups": 1},
                              {"n_groups": 3, 'eps': 1.0},
                              {"n_groups": 3, 'weights_shape': (6,), 'eps': -0.05},
                              {"n_groups": 3, 'weights_shape': (6,)},
                              {"n_groups": 2, 'weights_shape': (6,), 'bias': True},
                              {"n_groups": 2, 'weights_shape': (6,), 'bias': False},
                              {"n_groups": 2, 'weights_shape': (6,), 'bias': True, 'eps': 0.0},
                              {"n_groups": 2, 'weights_shape': (6,), 'bias': False, 'eps': 0.0001},
                             ])
    @pytest.mark.nightly
    def test_conv2d(self, params, ie_device, precision, ir_version):
        self._test(*self.create_model(**params),
                   ie_device, precision, ir_version)
