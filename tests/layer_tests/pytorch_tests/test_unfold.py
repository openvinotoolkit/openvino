# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytorch_layer_test_class import PytorchLayerTest
import numpy as np
import torch


# @pytest.mark.parametrize('dimension', (0, 1, 2))
# @pytest.mark.parametrize('size', (1, 2))
# @pytest.mark.parametrize('step', (1, 2))
# @pytest.mark.parametrize('input_tensor', (np.random.randn(2, 2, 5).astype(np.float32),
#                                           np.random.randn(3, 3, 3, 3).astype(np.float32)))
@pytest.mark.parametrize('step', (1,))
@pytest.mark.parametrize('size', (2,))
@pytest.mark.parametrize('dimension', (2,))
@pytest.mark.parametrize('input_tensor', (np.array([[[1, 2, 3, 4, 5],
                                                     [11, 12, 13, 14, 15]],
                                                    [[21, 22, 23, 24, 25],
                                                     [31, 32, 33, 34, 35]]]).astype(np.float32),))
class TestUnfold(PytorchLayerTest):

    def _prepare_input(self):
        return (self.input_tensor, )

    def create_model(self, dimension, size, step):
        class aten_unfold(torch.nn.Module):

            def __init__(self, dimension, size, step) -> None:
                super().__init__()
                self.dimension = dimension
                self.size = size
                self.step = step

            def forward(self, input_tensor):
                return input_tensor.unfold(dimension=self.dimension, size=self.size, step=self.step)

        ref_net = None

        return aten_unfold(dimension, size, step), ref_net, "aten::unfold"

    @pytest.mark.nightly
    def test_unfold(self, ie_device, precision, ir_version, dimension, size, step, input_tensor):
        self.input_tensor = input_tensor
        self._test(*self.create_model(dimension, size, step),
                   ie_device, precision, ir_version)
