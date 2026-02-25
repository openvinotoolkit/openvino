# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestNarrow(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(*self.input_shape).astype(np.float32),)

    def create_model(self, dim, start, length):

        class aten_narrow(torch.nn.Module):
            def __init__(self, dim, start, length):
                super().__init__()
                self.dim = dim
                self.start = start
                self.length = length

            def forward(self, input_tensor):
                return torch.narrow(input_tensor, dim=self.dim, start=self.start, length=self.length)

        return aten_narrow(dim, start, length), None, "aten::narrow"

    @pytest.mark.parametrize("input_shape", [
        [3, 3], [3, 4, 5]
    ])
    @pytest.mark.parametrize("dim", [0, 1, -1])
    @pytest.mark.parametrize("start", [0, 1])
    @pytest.mark.parametrize("length", [1, 2])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_narrow(self, input_shape, dim, start, length, ie_device, precision, ir_version):
        self.input_shape = input_shape
        self._test(*self.create_model(dim, start, length), ie_device, precision, ir_version)
