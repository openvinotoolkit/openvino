# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestScalarTensor(PytorchLayerTest):

    def _prepare_input(self):
        return (np.array(self.random.randn(), dtype=np.float32),)

    def create_model(self, dtype):
        class aten_scalar_tensor(torch.nn.Module):

            def __init__(self, dtype) -> None:
                super().__init__()
                self.dtype = dtype

            def forward(self, lhs):
                if self.dtype is None:
                    return torch.scalar_tensor(lhs.item())
                return torch.scalar_tensor(lhs.item(), dtype=self.dtype)


        return aten_scalar_tensor(dtype), f"aten::scalar_tensor"

    @pytest.mark.parametrize("dtype", [None, torch.float32, torch.float64, torch.int32, torch.int64])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_scalar_tensor(self, dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(dtype), ie_device, precision, ir_version, use_convert_model=True)


