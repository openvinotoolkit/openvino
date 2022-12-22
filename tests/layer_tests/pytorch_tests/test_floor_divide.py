# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest


class TestFloorDivide(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_tensor, self.other_tensor)

    def create_model(self):

        import torch

        class aten_floor_divide(torch.nn.Module):
            def __init__(self):
                super(aten_floor_divide, self).__init__()

            def forward(self, input_tensor, other_tensor):
                return torch.floor_divide(input_tensor, other_tensor)

        ref_net = None

        # return aten_floor_divide(), ref_net, "aten::floor_divide"
        return aten_floor_divide(), ref_net, "aten::floor_divide"

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
    @pytest.mark.nightly
    def test_floor_divide(self, input_tensor, other_tensor, ie_device, precision, ir_version):
        self.input_tensor = input_tensor 
        self.other_tensor = other_tensor 
        self._test(*self.create_model(), ie_device, precision, ir_version)
