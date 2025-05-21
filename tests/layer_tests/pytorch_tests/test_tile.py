# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestTile(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    def create_model(self, dims):
        import torch

        class aten_tile(torch.nn.Module):
            def __init__(self, dims):
                super(aten_tile, self).__init__()
                self.dims = dims

            def forward(self, x):
                return torch.tile(x, self.dims)

        ref_net = None

        return aten_tile(dims), ref_net, "aten::tile"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("dims", [(2, 2), (1, 1), (1, 2, 3, 4)])
    def test_tile(self, dims, ie_device, precision, ir_version):
        self._test(*self.create_model(dims), ie_device, precision, ir_version)
