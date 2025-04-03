# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest


class Reverse(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.flip(x, dims=self.dims)


class TestReverse(PytorchLayerTest):
    def _prepare_input(self, dtype="float32"):
        return (np.random.randn(1, 3, 32, 32).astype(dtype),)

    def create_model(self, dims):
        model = Reverse(dims)
        ref_net = None
        kind = "aten::flip"
        return model, ref_net, kind

    @pytest.mark.parametrize("dims", [
        [0], [1], [2], [1, 2], [-1], [-2], [0, 2], [0, 1, 2]
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_reverse(self, dims, ie_device, precision, ir_version):
        self._test(*self.create_model(dims),
                   ie_device,
                   precision,
                   ir_version)
