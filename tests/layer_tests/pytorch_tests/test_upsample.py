# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytorch_layer_test_class import PytorchLayerTest


class TestUpsample2D(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.zeros((1, 3, 224, 224)).astype(np.float32),)

    def create_model(self, size, scale, mode):

        import torch
        import torch.nn.functional as F

        class aten_upsample(torch.nn.Module):
            def __init__(self, size, scale, mode):
                super().__init__()
                self.size = size
                self.scale = scale
                self.mode = mode


            def forward(self, x):
                return F.interpolate(x, self.size, scale_factor=self.scale, mode=self.mode)
                

        ref_net = None

        return aten_upsample(size, scale, mode), ref_net, F"aten::upsample_{mode}2d"

    @pytest.mark.parametrize("mode,size,scale", [
        ('nearest', 300, None),
        ('nearest', 200, None), 
        ('nearest', (128, 480), None),
        ('nearest', None, 2.5,), 
        ('nearest', None, 0.75),
        ('nearest', None, (1.2, 0.8)),
        ('bilinear', 300, None),
        ('bilinear', 200, None), 
        ('bilinear', (128, 480), None),
        ('bilinear', None, 2.5,), 
        ('bilinear', None, 0.75),
        ('bilinear', None, (1.2, 0.8)),
        ('bicubic', 300, None),
        ('bicubic', 200, None), 
        ('bicubic', (128, 480), None),
        ('bicubic', None, 2.5,), 
        ('bicubic', None, 0.75),
        ('bicubic', None, (1.2, 0.8))]
    )
    @pytest.mark.nightly
    def test_upsample(self, mode, size, scale, ie_device, precision, ir_version):
        self._test(*self.create_model(size, scale, mode), ie_device, precision, ir_version, trace_model=True)