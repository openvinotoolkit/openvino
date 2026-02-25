# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


@pytest.mark.parametrize('input_tensor', (np.array([0.]), np.array([1.5]), np.array([False]), np.array([3])))
class TestIsNonZero(PytorchLayerTest):

    def _prepare_input(self):
        input_tensor = self.input_tensor
        return (input_tensor.astype(np.int64),)

    def create_model(self):
        class aten_is_nonzero(torch.nn.Module):

            def forward(self, input_tensor):
                return torch.is_nonzero(input_tensor)

        ref_net = None

        return aten_is_nonzero(), ref_net, "aten::is_nonzero"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_is_nonzero(self, ie_device, precision, ir_version, input_tensor):
        self.input_tensor = input_tensor
        self._test(*self.create_model(), ie_device, precision, ir_version)
