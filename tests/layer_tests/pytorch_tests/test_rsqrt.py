# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestRSqrt(PytorchLayerTest):
    def _prepare_input(self):
        return (torch.randn(1, 10).to(self.dtype).numpy(),)

    def create_model(self, dtype):
        class aten_rsqrt(torch.nn.Module):
            def __init__(self, dtype):
                super(aten_rsqrt, self).__init__()
                self.dtype = dtype

            def forward(self, x):
                return torch.rsqrt(x.to(self.dtype))

        ref_net = None

        return aten_rsqrt(dtype), ref_net, "aten::rsqrt"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int32, torch.int64])
    def test_relu(self, dtype, ie_device, precision, ir_version):
        self.dtype = dtype
        self._test(*self.create_model(dtype), ie_device, precision, ir_version)
