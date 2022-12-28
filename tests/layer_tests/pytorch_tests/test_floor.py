# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytorch_layer_test_class import PytorchLayerTest


class TestFloor(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    def create_model(self, inplace):
        import torch

        class aten_floor(torch.nn.Module):
            def __init__(self, inplace):
                super(aten_floor, self).__init__()
                self.op = torch.floor_ if inplace else torch.floor

            def forward(self, x):
                return x, self.op(x)

        ref_net = None

        return aten_floor(inplace), ref_net, "aten::floor" if not inplace else "aten::floor_"

    @pytest.mark.parametrize("inplace", [False, True])
    @pytest.mark.nightly
    def test_floor(self, inplace, ie_device, precision, ir_version):
        self._test(*self.create_model(inplace), ie_device, precision, ir_version)
