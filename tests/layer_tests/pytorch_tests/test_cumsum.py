# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestCumSum(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    def create_model(self, axis):
        import torch

        class aten_cumsum(torch.nn.Module):
            def __init__(self, axis):
                super(aten_cumsum, self).__init__()
                self.axis = axis

            def forward(self, x):
                return torch.cumsum(x, self.axis)

        ref_net = None

        return aten_cumsum(axis), ref_net, "aten::cumsum"

    @pytest.mark.parametrize("axis", [0, 1, 2, 3, -1, -2, -3, -4])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_cumsum(self, axis, ie_device, precision, ir_version):
        self._test(*self.create_model(axis), ie_device, precision, ir_version)
