# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


@pytest.mark.parametrize('test_input', [(np.array([[1, 2], [3, 4]], dtype=np.float32),
                                         np.array([[1, 1], [2, 2]], dtype=np.float32),),
                                        (np.array([[1, 2], [3, 4]], dtype=np.float32),
                                         np.array([2, 3], dtype=np.float32),),
                                        (np.array([[1, 2], [3, 4]], dtype=np.float32),
                                         np.array([2], dtype=np.float32),),
                                        (np.array([5, 6], dtype=np.float32),
                                         np.array([[1, 2], [3, 4]], dtype=np.float32),),
                                        (np.array([5], dtype=np.float32),
                                         np.array([[1, 2], [3, 4]], dtype=np.float32),)])
class TestPow(PytorchLayerTest):
    """
    Input test data contains five test cases - elementwise power, broadcast exponent, one exponent,
    broadcast base, one base.
    """

    def _prepare_input(self):
        return self.test_input

    def create_model(self):
        class aten_pow(torch.nn.Module):

            def forward(self, input_data, exponent):
                return torch.pow(input_data, exponent)

        ref_net = None

        return aten_pow(), ref_net, "aten::pow"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_pow(self, ie_device, precision, ir_version, test_input):
        self.test_input = test_input
        self._test(*self.create_model(), ie_device, precision, ir_version)
