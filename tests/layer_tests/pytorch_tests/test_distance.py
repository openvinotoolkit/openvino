# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestCdist(PytorchLayerTest):
    def _prepare_input(self, x_dtype="float32", y_dtype="float32"):
        import numpy as np
        return (np.random.randint(-10, 10, (2, 2)).astype(x_dtype), np.random.randint(-10, 10, (3, 2)).astype(y_dtype))

    def create_model(self, p):
        import torch

        class aten_cdist(torch.nn.Module):
            def __init__(self, p):
                super(aten_cdist, self).__init__()
                self.p = p

            def forward(self, x, y):
                return torch.cdist(x, y, self.p)

        ref_net = None

        return aten_cdist(p), ref_net, "aten::cdist"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("p", [2., 4., 6., 8.,])
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_cdist(self, p, ie_device, precision, ir_version):
        self._test(*self.create_model(p), ie_device, precision, ir_version, use_convert_model=True)


class TestPairwiseDistance(PytorchLayerTest):
    def _prepare_input(self, x_dtype="float32", y_dtype="float32"):
        import numpy as np
        return (np.random.randint(-10, 10, (20, 100)).astype(x_dtype), np.random.randint(-10, 10, (20, 100)).astype(y_dtype))

    def create_model(self, p, eps, keepdim):
        import torch
        import torch.nn.functional as F

        class aten_cdist(torch.nn.Module):
            def __init__(self, p, eps, keepdim):
                super(aten_cdist, self).__init__()
                self.p = p
                self.eps = eps
                self.keepdim = keepdim

            def forward(self, x, y):
                return F.pairwise_distance(x, y, self.p, self.eps, self.keepdim)

        ref_net = None

        return aten_cdist(p, eps, keepdim), ref_net, "aten::pairwise_distance"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("p", [2., 4., 6., 8.,])
    @pytest.mark.parametrize("eps", [1e-06, 0.00001, 1e-07])
    @pytest.mark.parametrize("keepdim", [True, False])
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_cdist(self, p, eps, keepdim, ie_device, precision, ir_version):
        self._test(*self.create_model(p, eps, keepdim), ie_device, precision, ir_version, use_convert_model=True)
