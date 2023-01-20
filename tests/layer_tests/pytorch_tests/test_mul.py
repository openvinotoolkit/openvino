# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestMul(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_array.astype(self.input_type), self.other_array.astype(self.other_type))

    def create_model(self):
        import torch

        class aten_mul(torch.nn.Module):
            def __init__(self):
                super(aten_mul, self).__init__()

            def forward(self, input_tensor, other_tensor):
                return torch.mul(input_tensor, other_tensor)

        ref_net = None

        return aten_mul(), ref_net, "aten::mul"

    @pytest.mark.parametrize(("input_array", "other_array"), [
        [np.random.rand(1, 2), np.random.rand(2, 1)],
        [np.random.rand(3, 1, 2), np.random.rand(3, 1, 2)],
        [np.random.rand(4, 1, 1), np.random.rand(1, 1, 4)],
    ])
    @pytest.mark.parametrize(("types"), [
        (np.float32, np.float32),
        # Type promotion
        pytest.param((np.int32, np.float32), marks=pytest.mark.xfail),
        pytest.param((np.float32, np.int32), marks=pytest.mark.xfail),
        pytest.param((np.int32, np.int32), marks=pytest.mark.xfail)
    ])
    @pytest.mark.nightly
    def test_mul_random(self, input_array, other_array, types, ie_device, precision, ir_version):
        self.input_array = input_array 
        self.input_type = types[0]
        self.other_array = other_array 
        self.other_type = types[1]
        self._test(*self.create_model(), ie_device, precision, ir_version)


    @pytest.mark.parametrize(("input_array", "other_array"), [
        [np.array([ 0.2015, -0.4255,  2.6087]), np.array(100)],
        [np.array([[ 1.1207], [-0.3137], [0.0700], [0.8378]]), np.array([[0.5146, 0.1216, -0.5244, 2.2382]])],
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_mul_pt_spec(self, input_array, other_array, ie_device, precision, ir_version):
        self.input_array = input_array 
        self.input_type = np.float32
        self.other_array = other_array
        self.other_type = np.float32 
        self._test(*self.create_model(), ie_device, precision, ir_version)
