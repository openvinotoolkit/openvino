# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestInplaceNormal(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    def create_model(self, mean, std):
        class aten_normal(torch.nn.Module):
            def __init__(self, mean, std):
                super(aten_normal, self).__init__()
                self.mean = mean
                self.std = std

            def forward(self, x):
                x = x.to(torch.float32)
                return x.normal_(mean=self.mean, std=self.std), x

        return aten_normal(mean, std), None, "aten::normal_"

    @pytest.mark.parametrize("mean,std", [(0., 1.), (5., 20.)])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_inplace_normal(self, mean, std, ie_device, precision, ir_version):
        self._test(*self.create_model(mean, std),
                   ie_device, precision, ir_version, custom_eps=1e30)


class TestNormal(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        if isinstance(self.inputs, list):
            return (np.random.randn(*self.inputs).astype(np.float32),)
        return self.inputs

    class aten_normal1(torch.nn.Module):
        def forward(self, mean, std):
            return torch.normal(mean, std)

    class aten_normal2(torch.nn.Module):
        def forward(self, mean, std):
            x = torch.empty_like(mean, dtype=torch.float32)
            return torch.normal(mean, std, out=x), x

    class aten_normal3(torch.nn.Module):
        def forward(self, mean):
            return torch.normal(mean)

    class aten_normal4(torch.nn.Module):
        def forward(self, mean):
            x = torch.empty_like(mean, dtype=torch.float32)
            return torch.normal(mean, out=x), x

    class aten_normal5(torch.nn.Module):
        def forward(self, mean):
            x = torch.empty_like(mean, dtype=torch.float32)
            return torch.normal(mean, 2., out=x), x

    class aten_normal6(torch.nn.Module):
        def forward(self, x):
            x = x.to(torch.float32)
            return torch.normal(0., 1., x.shape)

    class aten_normal7(torch.nn.Module):
        def forward(self, x):
            x = x.to(torch.float32)
            return torch.normal(0., 1., x.shape, out=x), x

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("model,inputs", [
        (aten_normal1(), (torch.arange(1., 11.).numpy(), torch.arange(1, 0, -0.1).numpy())),
        (aten_normal2(), (torch.arange(1., 11.).numpy(), torch.arange(1, 0, -0.1).numpy())),
        (aten_normal3(), (torch.arange(1., 11.).numpy(),)),
        (aten_normal4(), (torch.arange(1., 11.).numpy(),)),
        (aten_normal5(), (torch.arange(1., 11.).numpy(),)),
        (aten_normal6(), [1, 3, 224, 224]),
        (aten_normal7(), [1, 3, 224, 224]),
    ])
    def test_inplace_normal(self, model, inputs, ie_device, precision, ir_version):
        self.inputs = inputs
        self._test(model, None, "aten::normal",
                   ie_device, precision, ir_version, custom_eps=1e30)
