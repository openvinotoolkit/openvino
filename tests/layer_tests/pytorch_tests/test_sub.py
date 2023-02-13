# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestSub(PytorchLayerTest):
    def _prepare_input(self):
        return self.input_data

    def create_model(self):

        class aten_sub(torch.nn.Module):

            def forward(self, x, y, alpha: float):
                return torch.sub(x, y, alpha=alpha)

        ref_net = None

        return aten_sub(), ref_net, "aten::sub"

    @pytest.mark.parametrize('input_data', [(np.random.randn(2, 3, 4).astype(np.float32),
                                             np.random.randn(2, 3, 4).astype(np.float32),
                                             np.random.randn(1)),
                                            (np.random.randn(4, 2, 3).astype(np.float32),
                                             np.random.randn(1, 2, 3).astype(np.float32),
                                             np.random.randn(1)),])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_sub(self, ie_device, precision, ir_version, input_data):
        self.input_data = input_data
        self._test(*self.create_model(), ie_device, precision, ir_version)
