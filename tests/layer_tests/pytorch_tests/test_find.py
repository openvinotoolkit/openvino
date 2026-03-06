# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest

class TestFind(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randint(0, 2, self.input_shape).astype(self.input_dtype),)

    def create_model(self):
        class aten_find_model(torch.nn.Module):
            def forward(self, x):
                return torch.nonzero(x)

        ref_net = None
        return aten_find_model(), ref_net, "aten::nonzero"

    @pytest.mark.parametrize("input_shape", [
        [1, 10],
        [10, 1, 2],
        [2, 3, 4, 5]
    ])
    @pytest.mark.parametrize("input_dtype", [
        np.float32,
        np.int32,
        np.int64
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_find(self, ie_device, precision, ir_version, input_shape, input_dtype):
        self.input_shape = input_shape
        self.input_dtype = input_dtype
        self._test(*self.create_model(), ie_device, precision, ir_version)