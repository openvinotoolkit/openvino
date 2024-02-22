# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestUnsqueeze(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(5, 10).astype(np.float32),)

    def create_model(self, inplace=False, dim=0):
        import torch

        class aten_unsqueeze(torch.nn.Module):
            def __init__(self, dim):
                super(aten_unsqueeze, self).__init__()
                self.op = torch.unsqueeze
                self.dim = dim

            def forward(self, x):
                return x, self.op(x, self.dim)

        class aten_unsqueeze_(torch.nn.Module):
            def __init__(self, dim):
                super(aten_unsqueeze_, self).__init__()
                self.dim = dim

            def forward(self, x):
                return x, x.unsqueeze_(self.dim)

        ref_net = None
        model_class, op = (aten_unsqueeze, "aten::unsqueeze") if not inplace else (aten_unsqueeze_, "aten::unsqueeze_")

        return model_class(dim), ref_net, op

    @pytest.mark.parametrize("inplace", [False, True])
    @pytest.mark.parametrize("dim", [0, 1, -1])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_unsqueeze(self, inplace, dim, ie_device, precision, ir_version):
        self._test(*self.create_model(inplace, dim), ie_device, precision, ir_version)
