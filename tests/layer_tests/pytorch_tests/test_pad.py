# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytorch_layer_test_class import PytorchLayerTest


class TestPad(PytorchLayerTest):
    def _prepare_input(self, ndim=4):
        import numpy as np
        input_5d_shape = [1, 3, 14, 14, 18]
        return (np.random.randn(*input_5d_shape[:ndim]).astype(np.float32),)

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
    def test_pad4d(self, pads, mode, value, ie_device, precision, ir_version):
        self._test(*self.create_model(pads, mode, value), ie_device, precision, ir_version, kwargs_to_prepare_input={'ndim': 4})

    @pytest.mark.parametrize("pads,mode,value", [
    ((1, 2, 3, 4, 5, 6), "reflect", None),
    ((1, 0, 0, 0, 0, 1), "reflect", None),
    ((1, 0, 0, 0, 0, 0), "reflect", None),
    ((0, 0, 0, 0, 0, 0), "reflect", None),
    ((1, 2, 3, 4, 5, 6), "replicate", None),
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
    ((1, 0, 0, 0, 0, 1, 1, 2, 2, 3), "constant", 0.),
    ((1, 2, 0, 0, 0, 0), "circular", None),
    ((1, 2, 3, 4, 5, 6), "circular", None),
    ((0, 1, 0, 0, 0, 0), "circular", None),
    ((0, 0, 0, 0, 0, 0), "circular", None),
    ((0, 0, -1, -2, 0, 0), "circular", None),
    ((-1, -2, -1, -2, -1, -2), "circular", None),
    ((-5, -8, 0, 0, 0, 0), "circular", None),
    ((10, 10, 10, 10, 10, 10), "circular", None),
    ])
    @pytest.mark.nightly
    def test_pad5d(self, pads, mode, value, ie_device, precision, ir_version):
        self._test(*self.create_model(pads, mode, value), ie_device, precision, ir_version, kwargs_to_prepare_input={'ndim': 5}, trace_model=True)
    
    @pytest.mark.parametrize("pads,mode,value", [
    ((1, 2), "reflect", None),
    ((1, 0), "reflect", None),
    ((0, 0), "reflect", None),
    ((1, 2), "replicate", None),
    ((1, 0), "replicate", None),
    ((0, 0), "replicate", None),
    ((1, 0), "constant", None),
    ((1, 0), "constant", 42.),
    ((1, 0), "constant", -0.57),
    ((1, 2, 3, 4), "constant", None),
    ((1, 2, 3, 4), "constant", 42.),
    ((1, 2, 3, 4), "constant", -0.57),
    ])
    @pytest.mark.nightly
    def test_pad2d(self, pads, mode, value, ie_device, precision, ir_version):
        self._test(*self.create_model(pads, mode, value), ie_device, precision, ir_version, kwargs_to_prepare_input={'ndim': 2}, trace_model=True)
