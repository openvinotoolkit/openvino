# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest


class TestCopysign(PytorchLayerTest):
    def _prepare_input(self, input_shape=(3, 3), dtype=np.float32, is_scalar=False):
        lhs = self.random.uniform(-10.0, 10.0, size=input_shape).astype(dtype)
        if is_scalar:
            return (lhs,)
        rhs = self.random.uniform(-10.0, 10.0, size=input_shape).astype(dtype)
        return (lhs, rhs)

    def create_model(self, other_scalar=None):
        class aten_copysign(torch.nn.Module):
            def __init__(self, scalar=None):
                super().__init__()
                self.scalar = scalar

            def forward(self, x, y=None):
                if self.scalar is not None:
                    return torch.copysign(x, self.scalar)
                return torch.copysign(x, y)

        return aten_copysign(other_scalar), "aten::copysign"

    @pytest.mark.parametrize("input_shape", [(2, 3), (3, 3, 2)])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_copysign_tensor(self, input_shape, dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"input_shape": input_shape, "dtype": dtype, "is_scalar": False})

    @pytest.mark.parametrize("input_shape", [(3, 3)])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
    @pytest.mark.parametrize("scalar", [-1.5, 2.0, -0.5, -0.0, 0.0])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_copysign_scalar(self, input_shape, dtype, scalar, ie_device, precision, ir_version):
        self._test(*self.create_model(other_scalar=scalar), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"input_shape": input_shape, "dtype": dtype, "is_scalar": True})
