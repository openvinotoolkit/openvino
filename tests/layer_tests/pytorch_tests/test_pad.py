# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytorch_layer_test_class import PytorchLayerTest


class TestPad(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    def create_model(self, pads, mode, value=None):

        import torch
        import torch.nn.functional as F

        class aten_pad(torch.nn.Module):
            def __init__(self, pads, mode, value=None):
                super().__init__()
                self.pads = pads
                self.mode = mode
                self.value = value

            def forward(self, x):
                return F.pad(x, self.pads, mode=self.mode, value=self.value)

        ref_net = None

        return aten_pad(pads, mode, value), ref_net, "aten::pad"

    @pytest.mark.parametrize("pads,mode,value", [
        ((1, 2, 3, 4), "reflect", None),
        ((1, 0, 0, 0, 0, 1), "reflect", None),
        ((0, 0, 0, 0, 0, 0), "reflect", None),
        ((1, 2, 3, 4), "replicate", None),
        ((1, 0, 0, 0, 0, 0), "replicate", None),
        ((1, 0, 0, 0, 0, 1), "replicate", None),
        ((0, 0, 0, 0, 0, 0), "replicate", None),
        ((1, 2, 3, 4), "constant", None),
        ((1, 2, 3, 4), "constant", 42.),
        ((1, 2, 3, 4), "constant", -0.57),
        ((1, 2), "constant", None),
        ((1, 0, 0, 0, 0, 1), "constant", None),
        ((0, 0, 0, 0, 0, 0), "constant", None),
        ((1, 0, 0, 0, 0, 1, 1, 2), "constant", 0.),
        ((1, 2, 0, 0), "circular", None),
        ((1, 2, 3, 4), "circular", None),
        ((0, 1, 0, 0), "circular", None),
        ((0, 0, 0, 0), "circular", None),
        ((0, 0, -1, -2), "circular", None),
        ((-1, -2, -1, -2), "circular", None),
        ((-5, -8, 0, 0), "circular", None),
        ])
    @pytest.mark.nightly
    def test_pad(self, pads, mode, value, ie_device, precision, ir_version):
        self._test(*self.create_model(pads, mode, value), ie_device, precision, ir_version)