# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestStack2D(PytorchLayerTest):
    def _prepare_input(self):
        return self.input_tensors

    def create_model(self, dim):
        import torch

        class aten_stack(torch.nn.Module):
            def __init__(self, dim):
                super(aten_stack, self).__init__()
                self.dim = dim 

            def forward(self, x, y):
                inputs = [x, y]
                return torch.stack(inputs, self.dim)

        ref_net = None

        return aten_stack(dim), ref_net, "aten::stack"

    @pytest.mark.parametrize("input_tensor", ([
        [np.random.rand(1, 3, 3), np.random.rand(1, 3, 3)],
        [np.random.rand(4, 4, 2), np.random.rand(4, 4, 2)],
        [np.random.rand(8, 1, 1, 9), np.random.rand(8, 1, 1, 9)]
    ]))
    @pytest.mark.parametrize("dim", ([
        0, 1, 2,
    ]))
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_stack2D(self, input_tensor, dim, ie_device, precision, ir_version):
        self.input_tensors = input_tensor
        self._test(*self.create_model(dim), ie_device, precision, ir_version)


class TestStack3D(PytorchLayerTest):
    def _prepare_input(self):
        return self.input_tensors

    def create_model(self, dim):
        import torch

        class aten_stack(torch.nn.Module):
            def __init__(self, dim):
                super(aten_stack, self).__init__()
                self.dim = dim 

            def forward(self, x, y, z):
                inputs = [x, y, z]
                return torch.stack(inputs, self.dim)

        ref_net = None

        return aten_stack(dim), ref_net, "aten::stack"

    @pytest.mark.parametrize("input_tensor", ([
        [np.random.rand(1, 3, 3), np.random.rand(1, 3, 3), np.random.rand(1, 3, 3)],
        [np.random.rand(4, 4, 2), np.random.rand(4, 4, 2), np.random.rand(4, 4, 2)],
        [np.random.rand(8, 1, 1, 9), np.random.rand(8, 1, 1, 9), np.random.rand(8, 1, 1, 9)]
    ]))
    @pytest.mark.parametrize("dim", ([
        0, 1, 2,
    ]))
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_stack3D(self, input_tensor, dim, ie_device, precision, ir_version):
        self.input_tensors = input_tensor
        self._test(*self.create_model(dim), ie_device, precision, ir_version)
