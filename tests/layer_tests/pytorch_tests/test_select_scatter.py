# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytorch_layer_test_class import PytorchLayerTest
import torch


class TestSelectScatter(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(2, 5, 3, 4).astype(np.float32),)

    def create_model(self, src, dim, index):

        class aten_select_scatter(torch.nn.Module):
            def __init__(self, src=None, dim=None, index=None):
                super(aten_select_scatter, self).__init__()
                self.src = src
                self.dim = dim
                self.index = index

            def forward(self, x):
                return torch.select_scatter(x, self.src, self.dim, self.index);


        ref_net = None

        return aten_select_scatter(src, dim, index), ref_net, "aten::select_scatter"

    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize(("src", "dim", "index"),
                             [(torch.ones(2), 0, 0),])
    def aten_select_scatter(self, src, dim, index, ie_device, precision, ir_version):
        self._test(*self.create_model(src, dim, index),
                   ie_device, precision, ir_version)
