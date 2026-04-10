# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestInstanceNorm(PytorchLayerTest):
    def _prepare_input(self, ndim=4):
        shape5d = [3, 6, 10, 5, 2]
        shape = shape5d[:ndim]
        return (self.random.randn(*shape),)

    def create_model(self, weights=False, bias=False, mean_var=False, eps=1e-05):
        import torch

        class aten_instance_norm(torch.nn.Module):
            def __init__(self, rng, weights=False, bias=False, mean_var=False, eps=1e-05):
                super().__init__()
                weights_shape = (6, )
                self.weight = rng.torch_randn(*weights_shape) if weights else None
                self.bias = None
                self.use_input_stats = not mean_var
                if bias:
                    self.bias = rng.torch_randn(*weights_shape)
                self.mean = None
                self.var = None
                if mean_var:
                    self.mean = rng.torch_randn(*weights_shape)
                    self.var = rng.torch_randn(*weights_shape)

                self.eps = eps

            def forward(self, x):
                return torch.instance_norm(x, self.weight, self.bias, self.mean, self.var,  self.use_input_stats, 0.1, self.eps, False)


        return aten_instance_norm(self.random, weights, bias, mean_var, eps), "aten::instance_norm"

    @pytest.mark.parametrize("params",
                             [
                                 {"eps": 0.0001},
                                 {'weights': True, 'eps': -0.05},
                                 {'weights': True},
                                 {'weights': True, 'bias': True},
                                 {"weights": True, 'bias': False, "mean_var": True},
                                 {"weights": True, 'bias': True, "mean_var": True},
                                 {"weights": False, 'bias': True, "mean_var": True},
                                 {"weights": False, 'bias': False, "mean_var": True},
                                 {"weights": False, 'bias': False,
                                  "mean_var": True, "eps": 1.5}
                             ])
    @pytest.mark.parametrize("kwargs_to_prepare_input", [
        {"ndim": 3},
        {'ndim': 4},
        {"ndim": 5}
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_group_norm(self, params, ie_device, precision, ir_version, kwargs_to_prepare_input):
        self._test(*self.create_model(**params),
                   ie_device, precision, ir_version, kwargs_to_prepare_input=kwargs_to_prepare_input,
                   dynamic_shapes=not params.get("mean_var", False), use_mo_convert=False)
