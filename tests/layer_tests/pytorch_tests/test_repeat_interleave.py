# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytorch_layer_test_class import PytorchLayerTest
import numpy as np
import random
import torch


@pytest.mark.parametrize('input_data', ({'repeats': 1, 'dim': 0},
                                        {'repeats': 2, 'dim': 2},
                                        {'repeats': [2, 3], 'dim': 1},
                                        {'repeats': [3, 2, 1], 'dim': 3},
                                        {'repeats': 2, 'dim': None},
                                        {'repeats': [random.randint(1, 5) for _ in range(36)], 'dim': None}))
class TestRepeatInterleave(PytorchLayerTest):

    def _prepare_input(self):
        return (np.random.randn(2, 2, 3, 3), )

    def create_model(self, repeats, dim):
        class aten_repeat_interleave(torch.nn.Module):

            def __init__(self) -> None:
                super().__init__()
                self.repeats = torch.tensor(repeats, dtype=torch.int)
                self.dim = dim

            def forward(self, input_tensor):
                return input_tensor.repeat_interleave(self.repeats, self.dim)

        ref_net = None

        return aten_repeat_interleave(), ref_net, "aten::repeat_interleave"

    @pytest.mark.nightly
    def test_repeat_interleave(self, ie_device, precision, ir_version, input_data):
        repeats = input_data['repeats']
        dim = input_data['dim']
        self._test(*self.create_model(repeats, dim),
                   ie_device, precision, ir_version)
