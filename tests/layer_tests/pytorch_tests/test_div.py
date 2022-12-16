# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest


class Testdiv(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_tensor, self.other_tensor)

    def create_model(self, rounding_mode):

        import torch

        class aten_div(torch.nn.Module):
            def __init__(self, rounding_mode):
                super(aten_div, self).__init__()
                self.rounding_mode = rounding_mode

            def forward(self, input_tensor, other_tensor):
                return torch.div(input_tensor, other_tensor, rounding_mode=self.rounding_mode)

        ref_net = None

        return aten_div(rounding_mode), ref_net, "aten::div"

    @pytest.mark.parametrize('input_tensor', ([
        np.random.randn(5).astype(np.float32),
        np.random.randn(5, 5, 1).astype(np.float32),
        np.random.randn(1, 1, 5, 5).astype(np.float32),
    ]))
    @pytest.mark.parametrize('other_tensor', ([
        np.array([[0.5]]).astype(np.float32),
        np.random.randn(5).astype(np.float32),
        np.random.randn(5, 1).astype(np.float32),
        np.random.randn(1, 5).astype(np.float32),
    ]))
    @pytest.mark.parametrize('rounding_mode', ([
        None,
        "floor",
    ]))

    @pytest.mark.nightly
    def test_div(self, input_tensor, other_tensor, rounding_mode, ie_device, precision, ir_version):
        self.input_tensor = input_tensor 
        self.other_tensor = other_tensor 
        self._test(*self.create_model(rounding_mode), ie_device, precision, ir_version)

