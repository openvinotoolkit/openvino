# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytorch_layer_test_class import PytorchLayerTest
import numpy as np
import torch


@pytest.mark.parametrize('input_dim', list(range(4)))
@pytest.mark.parametrize('input_index', list(range(4)))
class TestSelect(PytorchLayerTest):

    def _prepare_input(self):
        return (np.random.randn(4, 4, 5, 5).astype(np.float32),)

    def create_model(self):

        input_dim = self.input_dim
        input_index = self.input_index

        class aten_select(torch.nn.Module):

            def __init__(self) -> None:
                super().__init__()
                nonlocal input_dim
                nonlocal input_index
                self.dim = torch.from_numpy(
                    np.array([input_dim], dtype=np.int32))
                self.index = torch.from_numpy(
                    np.array([input_index], dtype=np.int32))

            def forward(self, input_tensor):
                return torch.select(input_tensor, int(self.dim), int(self.index))

        ref_net = None

        return aten_select(), ref_net, "aten::select"

    @pytest.mark.nightly
    def test_pow(self, ie_device, precision, ir_version, input_dim, input_index):
        self.input_dim = input_dim
        self.input_index = input_index
        self._test(*self.create_model(), ie_device, precision, ir_version)
