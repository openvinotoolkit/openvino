# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestBitwiseNot(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return ((np.random.randn(1, 5) > 0).astype(bool),)

    def create_model(self):
        import torch

        class aten_bitwise_not(torch.nn.Module):

            def forward(self, x):
                return torch.bitwise_not(x)

        ref_net = None

        return aten_bitwise_not(), ref_net, "aten::bitwise_not"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_bitwise_not(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)