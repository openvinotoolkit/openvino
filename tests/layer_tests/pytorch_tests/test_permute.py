# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestPermute(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    def create_model(self, order):
        import torch

        class aten_permute(torch.nn.Module):
            def __init__(self, order):
                super(aten_permute, self).__init__()
                self.order = order

            def forward(self, x):
                return torch.permute(x, self.order)

        ref_net = None

        return aten_permute(order), ref_net, "aten::permute"

    @pytest.mark.parametrize("order", [[0, 2, 3, 1], [0, 3, 1, 2]])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_relu(self, order, ie_device, precision, ir_version):
        self._test(*self.create_model(order), ie_device, precision, ir_version)
