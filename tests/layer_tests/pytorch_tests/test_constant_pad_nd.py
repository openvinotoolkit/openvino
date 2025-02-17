# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestConstantPadND(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(2, 5, 3, 4).astype(np.float32),)

    def create_model(self, pad, value):

        import torch
        class aten_constant_pad_nd(torch.nn.Module):
            def __init__(self, pad=None, value=None):
                super(aten_constant_pad_nd, self).__init__()
                self.pad = pad
                self.value = value

            def forward(self, x):
                return torch.constant_pad_nd(x, self.pad, self.value);


        ref_net = None

        return aten_constant_pad_nd(pad, value), ref_net, "aten::constant_pad_nd"

    @pytest.mark.parametrize(("pad", "value"),
                             [((1,1,1,1), 0),((0,2,0,2), -1.0),((3,1,5,2), 0.5),((0,0,0,0), 0),])

    @pytest.mark.precommit_fx_backend
    def test_constant_pad_nd(self, pad, value, ie_device, precision, ir_version):
        self._test(*self.create_model(pad, value),
                   ie_device, precision, ir_version)
