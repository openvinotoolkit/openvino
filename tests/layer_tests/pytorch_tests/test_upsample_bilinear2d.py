# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytorch_layer_test_class import PytorchLayerTest


class TestUpsampleBilinear2d(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    def create_model(self, size, scale):

        import torch

        class aten_upsample_bilinear2d(torch.nn.Module):
            def __init__(self, size, scale):
                super().__init__()
                self.op = torch.nn.UpsamplingBilinear2d(size, scale_factor=scale)
            def forward(self, x):
                return self.op(x)
                

        ref_net = None

        return aten_upsample_bilinear2d(size, scale), ref_net, "aten::upsample_bilinear2d"

    @pytest.mark.parametrize("size,scale", [
        (300, None),
        (200, None), 
        ((128, 480), None),
        (None, 2.5,), 
        (None, 0.75),
        (None, (1.2, 0.8))]
        )
    @pytest.mark.nightly
    def test_upsample_bilinear2d(self, size, scale, ie_device, precision, ir_version):
        self._test(*self.create_model(size, scale), ie_device, precision, ir_version, trace_model=True)