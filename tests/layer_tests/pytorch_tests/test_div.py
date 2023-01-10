# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest


class Testdiv(PytorchLayerTest):
    def _prepare_input(self):
        # return (self.input_tensor, self.other_tensor)
        return (np.random.rand(*self.input_shape).astype(self.input_type),
                np.random.rand(*self.other_shape).astype(self.other_type))

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

    @pytest.mark.parametrize(("input_shape", "input_type"), [
        [(5, 5), np.float32],
        [(5, 5, 1), np.float32],
        [(1, 1, 5, 5), np.float32],

    ])
    @pytest.mark.parametrize(("other_shape", "other_type"), [
        [(1, 1), np.int32],
        [(5, 1), np.int32],
        [(1, 5), np.int32],
        [(1, 1), np.float32],
        [(5, 1), np.float32],
        [(1, 5), np.float32],
    ])
    @pytest.mark.parametrize('rounding_mode', ([
        None,
        "floor",
        "trunc"
    ]))

    @pytest.mark.nightly
    def test_div(self, input_shape, input_type, other_shape, other_type, rounding_mode, ie_device, precision, ir_version):
        self.input_shape = input_shape
        self.input_type = input_type
        self.other_shape = other_shape
        self.other_type = other_type
        self._test(*self.create_model(rounding_mode), ie_device, precision, ir_version)
