# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest


class TestDiv(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_array.astype(self.input_type), self.other_array.astype(self.other_type))

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

    @pytest.mark.parametrize(("input_array", "other_array"), [
        [10 * np.random.rand(5, 5), np.random.uniform(low=1, high=5, size=(1))],
        [10 * np.random.rand(5, 5, 1), np.random.uniform(low=1, high=5, size=(1))],
        [10 * np.random.rand(1, 1, 5, 5), np.random.uniform(
            low=1, high=5, size=(1))],
        [10 * np.random.rand(5, 5, 1), np.random.uniform(
            low=1, high=5, size=(5, 1))]
    ])
    @pytest.mark.parametrize(("types"), [
        (np.float32, np.float32),
        pytest.param((np.int32, np.float32), marks=pytest.mark.xfail),
        pytest.param((np.float32, np.int32), marks=pytest.mark.xfail),
        pytest.param((np.int32, np.int32), marks=pytest.mark.xfail)
    ])
    @pytest.mark.parametrize('rounding_mode', ([
        None,
        "floor",
        "trunc"
    ]))
    @pytest.mark.nightly
    def test_div(self, input_array, other_array, types, rounding_mode, ie_device, precision, ir_version):
        self.input_array = input_array
        self.input_type = types[0]
        self.other_array = other_array
        self.other_type = types[1]
        self._test(*self.create_model(rounding_mode),
                   ie_device, precision, ir_version)

    @pytest.mark.parametrize(("input_array", "other_array"), [
        [np.array([0.7620,  2.5548, -0.5944, -0.7438,  0.9274]), np.array(0.5)],
        [np.array([[-0.3711, -1.9353, -0.4605, -0.2917],
                   [0.1815, -1.0111,  0.9805, -1.5923],
                   [0.1062,  1.4581,  0.7759, -1.2344],
                   [-0.1830, -0.0313,  1.1908, -1.4757]]),
         np.array([0.8032,  0.2930, -0.8113, -0.2308])]
    ])
    @pytest.mark.parametrize('rounding_mode', ([
        None,
        "floor",
        "trunc"
    ]))
    @pytest.mark.nightly
    def test_div_pt_spec(self, input_array, other_array, rounding_mode, ie_device, precision, ir_version):
        self.input_array = input_array
        self.input_type = np.float32
        self.other_array = other_array
        self.other_type = np.float32
        self._test(*self.create_model(rounding_mode),
                   ie_device, precision, ir_version)
