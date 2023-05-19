# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pytorch_layer_test_class import PytorchLayerTest

class aten_log_softmax(torch.nn.Module):
    def __init__(self, dim, dtype) -> None:
        super().__init__()
        self.dim = dim
        self.dtype = dtype

    def forward(self, input_tensor):
        return F.log_softmax(input_tensor, dim = self.dim, dtype = self.dtype)

class TestLogSoftmax(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_tensor,)

    @pytest.mark.parametrize(["input_tensor", "dtype"], [
        [np.random.randint(-100, 100, (5, 9, 7)), torch.float32],
        [np.random.randint(-100, 100, (5, 9, 7)), torch.float64],
        [np.random.randn(10, 13, 11), None],
        [np.random.randn(10, 13, 11), torch.float32]
    ])
    @pytest.mark.parametrize("dim", [
        0,
        1, 
        -1
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_log_softmax(self, input_tensor, dim, dtype, ie_device, precision, ir_version):
        self.input_tensor = input_tensor
        self._test(aten_log_softmax(dim, dtype), None, "aten::log_softmax", 
                    ie_device, precision, ir_version)
