# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from pytorch_layer_test_class import PytorchLayerTest

class aten_log_softmax(torch.nn.Module):
    def __init__(self, dim, dtype) -> None:
        super().__init__()
        self.dim = dim
        self.dtype = dtype

    def forward(self, input_tensor):
        return F.log_softmax(input_tensor, dim = self.dim, dtype = self.dtype)

class TestLogSoftmax(PytorchLayerTest):
    def _prepare_input(self):
        if self.input_dtype == torch.float:
            self.input_tensor = self.random.randn(5, 9, 7)
        else:
            self.input_tensor = self.random.randint(-100, 100, (5, 9, 7))
        return (self.input_tensor,)

    @pytest.mark.parametrize(["input_dtype", "convert_dtype"], [
        # convert_dtype cannot be of type int from pytorch limitations
        [torch.int,   torch.float32],
        [torch.int,   torch.float64],
        [torch.float, None],
        [torch.float, torch.float64]
    ])
    @pytest.mark.parametrize("dim", [
        0,
        1,
        -1
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_log_softmax(self, input_dtype, convert_dtype, dim, ie_device, precision, ir_version):
        self.input_dtype = input_dtype
        self._test(aten_log_softmax(dim, convert_dtype), "aten::log_softmax",
                    ie_device, precision, ir_version)


class aten_log_softmax_scalar(torch.nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input_tensor):
        return F.log_softmax(input_tensor, dim=self.dim)


class TestLogSoftmaxScalar(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(1).reshape(()).astype(np.float32),)

    @pytest.mark.parametrize("dim", [-1, 0])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_log_softmax_scalar(self, dim, ie_device, precision, ir_version):
        self._test(aten_log_softmax_scalar(dim), None, "aten::log_softmax",
                   ie_device, precision, ir_version)
