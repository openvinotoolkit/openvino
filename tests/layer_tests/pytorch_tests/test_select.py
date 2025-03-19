# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


@pytest.mark.parametrize('input_dim', list(range(-3, 4)))
@pytest.mark.parametrize('input_index', list(range(-3, 4)))
class TestSelect(PytorchLayerTest):

    def _prepare_input(self):
        return (np.random.randn(4, 4, 5, 5).astype(np.float32),)

    def create_model(self, input_dim, input_index):
        class aten_select(torch.nn.Module):

            def __init__(self, input_dim, input_index) -> None:
                super().__init__()
                self.dim = input_dim
                self.index = input_index

            def forward(self, input_tensor):
                return torch.select(input_tensor, int(self.dim), int(self.index))

        ref_net = None

        return aten_select(input_dim, input_index), ref_net, "aten::select"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_select(self, ie_device, precision, ir_version, input_dim, input_index):
        self._test(*self.create_model(input_dim, input_index),
                   ie_device, precision, ir_version)

@pytest.mark.parametrize('input_dim', list(range(-3, 4)))
@pytest.mark.parametrize('input_index', list(range(-3, 4)))
class TestSelectCopy(PytorchLayerTest):

    def _prepare_input(self):
        return (np.random.randn(4, 4, 5, 5).astype(np.float32),)

    def create_model(self, input_dim, input_index):
        class aten_select_copy(torch.nn.Module):

            def __init__(self, input_dim, input_index) -> None:
                super().__init__()
                self.dim = input_dim
                self.index = input_index

            def forward(self, input_tensor):
                return torch.select_copy(input_tensor, int(self.dim), int(self.index))

        ref_net = None

        return aten_select_copy(input_dim, input_index), ref_net, "aten::select_copy"

    @pytest.mark.precommit_fx_backend
    def test_select_copy(self, ie_device, precision, ir_version, input_dim, input_index):
        self._test(*self.create_model(input_dim, input_index),
                   ie_device, precision, ir_version)
