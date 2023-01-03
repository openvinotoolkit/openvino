# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytorch_layer_test_class import PytorchLayerTest
import numpy as np
import torch


@pytest.mark.parametrize('input_tensor', (np.random.randn(2, 1, 3), np.random.randn(3, 7),
                                          np.random.randn(1, 1, 4, 4)))
class TestLen(PytorchLayerTest):

    def _prepare_input(self):
        input_tensor = self.input_tensor*10
        return (input_tensor.astype(np.int64), )

    def create_model(self):
        class aten_len(torch.nn.Module):

            def forward(self, input_tensor):
                return torch.as_tensor(len(input_tensor), dtype=torch.int)

        ref_net = None

        return aten_len(), ref_net, "aten::len"

    @pytest.mark.nightly
    def test_len(self, ie_device, precision, ir_version, input_tensor):
        self.input_tensor = input_tensor
        self._test(*self.create_model(), ie_device, precision, ir_version)
