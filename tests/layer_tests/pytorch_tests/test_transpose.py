# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestTranspose(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(2, 3, 4, 5).astype(np.float32),)

    def create_model(self, dim0, dim1):
        import torch

        class aten_transpose(torch.nn.Module):
            def __init__(self, dim0, dim1):
                super(aten_transpose, self).__init__()
                self.dim0 = dim0
                self.dim1 = dim1

            def forward(self, x):
                return torch.transpose(x, self.dim0, self.dim1)

        ref_net = None

        return aten_transpose(dim0, dim1), ref_net, "aten::transpose"

    @pytest.mark.parametrize("dim0", [0, 1, 2, 3, -1, -2, -3, -4])
    @pytest.mark.parametrize("dim1", [0, 1, 2, 3, -1, -2, -3, -4])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_relu(self, dim0, dim1, ie_device, precision, ir_version):
        self._test(*self.create_model(dim0, dim1),
                   ie_device, precision, ir_version)
