# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestSqueeze(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np

        return (np.random.randn(1, 1, 32).astype(np.float32),)

    def create_model(self, dim):
        import torch

        class aten_squeeze(torch.nn.Module):
            def __init__(self, dim):
                super(aten_squeeze, self).__init__()
                self.dim = dim

            def forward(self, x):
                if self.dim is not None:
                    return torch.squeeze(x, self.dim)
                return torch.squeeze(x)

        ref_net = None

        return aten_squeeze(dim), ref_net, "aten::squeeze"

    @pytest.mark.parametrize("dim", [-2, -1, 0, None, [-1, -2], 1, 2])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_squeeze(self, dim, ie_device, precision, ir_version):
        # Dynamic shapes are introducing dynamic rank, with is not suppoerted by Squeeze operation.
        self._test(*self.create_model(dim), ie_device, precision, ir_version, dynamic_shapes=False)
