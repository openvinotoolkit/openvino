# Copyright (C) 2018-2025 Intel Corporation
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

    @pytest.mark.parametrize("input_shape",
    [
        [1, 3, 3],
        [4, 4, 2],
        [8, 1, 1, 9]
    ])
    @pytest.mark.parametrize("dim", ([
        0, 1, 2,
    ]))
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_stack2D(self, input_shape, dim, ie_device, precision, ir_version):
        self.input_tensors = [
            np.random.randn(*input_shape).astype(np.float32),
            np.random.randn(*input_shape).astype(np.float32),
        ]
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

    @pytest.mark.parametrize("input_shape",
    [
        [1, 3, 3],
        [4, 4, 2],
        [8, 1, 1, 9]
    ])
    @pytest.mark.parametrize("dim", ([
        0, 1, 2,
    ]))
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_stack3D(self, input_shape, dim, ie_device, precision, ir_version):
        self.input_tensors = [
            np.random.randn(*input_shape).astype(np.float32),
            np.random.randn(*input_shape).astype(np.float32),
            np.random.randn(*input_shape).astype(np.float32)
        ]
        self._test(*self.create_model(dim), ie_device, precision, ir_version)
