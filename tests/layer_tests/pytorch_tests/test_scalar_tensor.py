# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestScalarTensor(PytorchLayerTest):

    def _prepare_input(self):
        return (np.array(self.random.randn(), dtype=np.float32),)

    def create_model(self):
        class aten_scalar_tensor(torch.nn.Module):

            def __init__(self) -> None:
                super().__init__()

            def forward(self, lhs):
                return torch.scalar_tensor(lhs.item())


        return aten_scalar_tensor(), f"aten::scalar_tensor"

    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_scalar_tensor(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version, use_convert_model=True)


class TestZeroDimBF16Buffer:
    """Regression test: 0-dim bfloat16/float8 buffers must not crash
    torch_tensor_to_ov_const during model conversion."""

    @staticmethod
    def _build_params():
        params = [torch.bfloat16]
        f8_e4m3 = getattr(torch, "float8_e4m3fn", None)
        if f8_e4m3 is not None:
            params.append(f8_e4m3)
        f8_e5m2 = getattr(torch, "float8_e5m2", None)
        if f8_e5m2 is not None:
            params.append(f8_e5m2)
        return params

    @pytest.mark.precommit
    @pytest.mark.parametrize("buf_dtype", _build_params.__func__())
    def test_zero_dim_buffer_conversion(self, buf_dtype):
        from openvino.frontend.pytorch.utils import torch_tensor_to_ov_const

        if buf_dtype == torch.bfloat16:
            t = torch.tensor(8.0, dtype=torch.bfloat16)
        else:
            t = torch.zeros(1, dtype=buf_dtype).reshape(())

        const = torch_tensor_to_ov_const(t, shared_memory=False)
        assert const.get_output_shape(0) == [1]


