# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestGroupNorm(PytorchLayerTest):
    def _prepare_input(self, ndim=4):
        shape5d = [20, 6, 10, 10, 10]
        shape = shape5d[:ndim]
        return (self.random.randn(*shape),)

    def create_model(self, n_groups, weights_shape=None, bias=False, eps=1e-05):
        import torch
        import torch.nn.functional as F

        class aten_group_norm(torch.nn.Module):
            def __init__(self, rng, n_groups, weights_shape=None, bias=True, eps=1e-05):
                super().__init__()
                self.weight = rng.torch_randn(*weights_shape) if weights_shape else None
                self.bias = None
                if bias:
                    self.bias = rng.torch_randn(*weights_shape)
                self.n_groups = n_groups
                self.eps = eps

            def forward(self, x):
                return F.group_norm(x, self.n_groups, self.weight, self.bias, self.eps)


        return aten_group_norm(self.random, n_groups, weights_shape, bias, eps), "aten::group_norm"

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
    @pytest.mark.parametrize("kwargs_to_prepare_input", [
        {"ndim": 3},
        {'ndim': 4},
        {"ndim": 5}
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_group_norm(self, params, ie_device, precision, ir_version, kwargs_to_prepare_input):
        self._test(*self.create_model(**params),
                   ie_device, precision, ir_version, kwargs_to_prepare_input=kwargs_to_prepare_input)
