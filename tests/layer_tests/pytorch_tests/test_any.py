# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestAny(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return ((np.random.randint(2, size=(3,3,10,10)) > 0),)

    def create_model(self, dim, keep_dim):

        import torch
        class aten_any(torch.nn.Module):
            def __init__(self, dim=None, keep_dim=None):
                super(aten_any, self).__init__()
                self.dim = dim
                self.keep_dim = keep_dim

            def forward(self, x):
                return torch.any(x, dim=self.dim, keepdim=self.keep_dim)


        ref_net = None

        return aten_any(dim, keep_dim), ref_net, "aten::any"

    @pytest.mark.parametrize(("dim", "keep_dim"),
                             [(0, False), (0, True), (-1, True)])

    @pytest.mark.precommit_fx_backend
    def test_any(self, dim, keep_dim, ie_device, precision, ir_version):
        self._test(*self.create_model(dim, keep_dim),
                   ie_device, precision, ir_version)
