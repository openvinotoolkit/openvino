# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytorch_layer_test_class import PytorchLayerTest


class TestBatchNorm(PytorchLayerTest):
    def _prepare_input(self, ndim=4):
        shape5d = [20, 6, 10, 10, 10]
        shape = shape5d[:ndim]
        return (self.random.randn(*shape),)

    def create_model(self, weights, bias, eps, train, running_stats):

        import torch
        import torch.nn.functional as F

        class aten_batch_norm_inference(torch.nn.Module):
            def __init__(self, rng, weights=True, bias=True, eps=1e-05):
                super().__init__()
                self.weight = rng.torch_randn(6) if weights else None
                self.bias = rng.torch_randn(6) if bias else None
                self.running_mean = rng.torch_randn(6)
                self.running_var = rng.torch_randn(6)
                self.eps = eps

            def forward(self, x):
                return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, eps=self.eps, training=False)

        class aten_batch_norm_train(torch.nn.Module):
            def __init__(self, rng, weights=True, bias=True, eps=1e-05, running_stats=False):
                super().__init__()
                self.weight = rng.torch_randn(6) if weights else None
                self.bias = rng.torch_randn(6) if bias else None
                self.running_mean = rng.torch_randn(6) if running_stats else None
                self.running_var = rng.torch_randn(6) if running_stats else None
                self.eps = eps

            def forward(self, x):
                return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, eps=self.eps, training=True)


        return aten_batch_norm_inference(self.random, weights, bias, eps) if not train else aten_batch_norm_train(self.random, weights, bias, eps, running_stats), "aten::batch_norm"

    @pytest.mark.parametrize("weights", [True, False])
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("eps", [1.0, 0.00005, 0.5, 0.042])
    @pytest.mark.parametrize(("train", "running_stats"), [(True, False), (True, True), (False, False)])
    @pytest.mark.parametrize("kwargs_to_prepare_input",
     [
        {"ndim": 3},
        {'ndim': 4},
        {"ndim": 5}
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    @pytest.mark.precommit_torch_export
    def test_batch_norm(self, weights, bias, eps, train, running_stats, ie_device, precision, ir_version, kwargs_to_prepare_input):
        if running_stats and self.use_torch_export():
            pytest.skip("running_mean not supported by torch.export")
        self._test(*self.create_model(weights, bias, eps, train, running_stats),
                   ie_device, precision, ir_version, kwargs_to_prepare_input=kwargs_to_prepare_input, dynamic_shapes=False, use_mo_convert=False)
