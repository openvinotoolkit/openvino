# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytorch_layer_test_class import PytorchLayerTest


class TestRelu(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(2, 10).astype(np.float32),)

    def create_model(self, repeats):

        import torch
        import torch.nn.functional as F

        class aten_repeat(torch.nn.Module):
            def __init__(self, repeats):
                super(aten_repeat, self).__init__()
                self.repeats = repeats

            def forward(self, x):
                return x.repeat(self.repeats)

        ref_net = None

        return aten_repeat(repeats), ref_net, "aten::repeat"

    @pytest.mark.parametrize("repeats", [(4, 3), (1, 1), (1, 2, 3), (1, 2, 2, 3)])
    @pytest.mark.nightly
    def test_relu(self, repeats, ie_device, precision, ir_version):
        self._test(*self.create_model(repeats), ie_device, precision, ir_version)