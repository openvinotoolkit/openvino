# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestStack2D(PytorchLayerTest):
    def _prepare_input(self):
        return self.input_tensors

    def create_model(self, dim=None):
        import torch

        class aten_stack(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, x, y):
                inputs = [x, y]
                if self.dim is None:
                    return torch.stack(inputs)
                return torch.stack(inputs, self.dim)


        return aten_stack(dim), "aten::stack"

    @pytest.mark.parametrize("input_shape",
    [
        [1, 3, 3],
        [4, 4, 2],
        [8, 1, 1, 9]
    ])
    @pytest.mark.parametrize("dim", ([
        0, 1, 2, None
    ]))
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_stack2D(self, input_shape, dim, ie_device, precision, ir_version):
        self.input_tensors = [
            self.random.randn(*input_shape),
            self.random.randn(*input_shape),
        ]
        self._test(*self.create_model(dim), ie_device, precision, ir_version)


class TestStack3D(PytorchLayerTest):
    def _prepare_input(self):
        return self.input_tensors

    def create_model(self, dim):
        import torch

        class aten_stack(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, x, y, z):
                inputs = [x, y, z]
                if self.dim is None:
                    return torch.stack(inputs)
                return torch.stack(inputs, self.dim)


        return aten_stack(dim), "aten::stack"

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
    @pytest.mark.precommit_fx_backend
    def test_stack3D(self, input_shape, dim, ie_device, precision, ir_version):
        self.input_tensors = [
            self.random.randn(*input_shape),
            self.random.randn(*input_shape),
            self.random.randn(*input_shape)
        ]
        self._test(*self.create_model(dim), ie_device, precision, ir_version)
