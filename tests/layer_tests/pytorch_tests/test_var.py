# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestVar(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    def create_model(self, unbiased, dim=None, keepdim=True, two_args_case=True):
        import torch

        class aten_var(torch.nn.Module):
            def __init__(self, dim, unbiased, keepdim):
                super(aten_var, self).__init__()
                self.unbiased = unbiased
                self.dim = dim
                self.keepdim = keepdim

            def forward(self, x):
                return torch.var(x, self.dim, unbiased=self.unbiased, keepdim=self.keepdim)

        class aten_var2args(torch.nn.Module):
            def __init__(self, unbiased):
                super(aten_var2args, self).__init__()
                self.unbiased = unbiased

            def forward(self, x):
                return torch.var(x, self.unbiased)

        ref_net = None
        if two_args_case:
            return aten_var2args(unbiased), ref_net, "aten::var"
        return aten_var(dim, unbiased, keepdim), ref_net, "aten::var"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("unbiased", [True, False])
    def test_var2args(self, unbiased, ie_device, precision, ir_version):
        self._test(*self.create_model(unbiased), ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("unbiased", [False, True])
    @pytest.mark.parametrize("dim", [None, 0, 1, 2, 3, -1, -2, (0, 1), (-1, -2), (0, 1, -1), (0, 1, 2, 3)])
    @pytest.mark.parametrize("keepdim", [True, False])
    def test_var(self, unbiased, dim, keepdim, ie_device, precision, ir_version):
        self._test(*self.create_model(unbiased, dim, keepdim, two_args_case=False), ie_device, precision, ir_version)
