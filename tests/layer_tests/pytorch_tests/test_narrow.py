# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestNarrow(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_tensor,)

    def create_model(self, dim, start, length):
        import torch

        class aten_narrow(torch.nn.Module):
            def __init__(self, dim, start, length):
                super(aten_narrow, self).__init__()
                self.dim = dim
                self.start = start
                self.length = length 

            def forward(self, x):
                return torch.narrow(x, dim=self.dim, start=self.start, length=self.length)

        ref_net = None

        return aten_narrow(dim, start, length), ref_net, "aten::narrow"

    @pytest.mark.parametrize("input_tensor", [
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[5, 2, 4], [4, 9, 0], [7, 7, 9]], [[5, 2, 4], [4, 9, 0], [7, 7, 9]]])
    ])
    @pytest.mark.parametrize("dim", [
        0, 1, -1
    ])
    @pytest.mark.parametrize("start", [
        0, 1
    ])
    @pytest.mark.parametrize("length", [
        1, 2,
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_narrow(self, input_tensor, dim, start, length, ie_device, precision, ir_version):
        self.input_tensor = input_tensor 
        if ie_device == "CPU":
            self._test(*self.create_model(dim, start, length), ie_device, precision, ir_version)

