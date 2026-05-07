# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from pytorch_layer_test_class import PytorchLayerTest, skip_if_export


class aten_rrelu(torch.nn.Module):
    def __init__(self, lower, upper, dtype, inplace):
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.dtype = dtype
        self.inplace = inplace

    def forward(self, x):
        x = x.to(self.dtype)
        return F.rrelu(x, lower=self.lower, upper=self.upper, training=False, inplace=self.inplace), x


class TestRRelu(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(2, 3, 4),)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("lower,upper", [
        (1 / 8, 1 / 3),   # PyTorch defaults
        (0.1, 0.5),
        (0.0, 1.0),
    ])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
    @pytest.mark.parametrize("inplace", [skip_if_export(True), False])
    def test_rrelu(self, lower, upper, dtype, inplace, ie_device, precision, ir_version):
        kwargs = {}
        if dtype == torch.float16:
            kwargs["custom_eps"] = 1e-2
        self._test(
            aten_rrelu(lower, upper, dtype, inplace),
            "aten::rrelu_" if inplace else "aten::rrelu",
            ie_device,
            precision,
            ir_version,
            **kwargs,
        )
