# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestArgSort(PytorchLayerTest):

    def _prepare_input(self):
        return (self.input_tensor,)

    def create_model(self, dim, descending, stable):
        class aten_argsort(torch.nn.Module):
            def __init__(self, dim, descending, stable) -> None:
                torch.nn.Module.__init__(self)
                self.dim = dim
                self.descending = descending
                self.stable = stable

            def forward(self, input_tensor):
                return torch.argsort(input_tensor, dim = self.dim, descending = self.descending, stable = self.stable)
        ref_net = None

        return aten_argsort(dim, descending, stable), ref_net, "aten::argsort"

    @pytest.mark.parametrize("input_tensor", [
        np.random.rand(1, 4),
        np.random.rand(4, 4),
        np.random.rand(4, 4, 4),
        np.random.rand(1, 2, 3, 4, 5),
    ])
    @pytest.mark.parametrize("dim", [
        0,
        1,
        -1
    ])
    @pytest.mark.parametrize("descending", [
        True,
        False,
    ])
    # Unused
    @pytest.mark.parametrize("stable", [
        False
    ])
    def test_argsort(self, input_tensor, dim, descending, stable, ie_device, precision, ir_version):
        self.input_tensor = input_tensor
        self._test(*self.create_model(dim, descending, stable), ie_device, precision, ir_version)
