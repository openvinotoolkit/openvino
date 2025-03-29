# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from pytorch_layer_test_class import PytorchLayerTest, skip_if_export


class aten_celu(torch.nn.Module):
    def __init__(self, alpha, dtype, inplace):
        super(aten_celu, self).__init__()
        self.alpha = alpha
        self.dtype = dtype
        self.inplace = inplace
        if alpha is None:
            self.forward = self.forward_no_alpha

    def forward(self, x):
        x_copy = x.to(self.dtype)
        return F.celu(x_copy, alpha=self.alpha, inplace=self.inplace), x_copy

    def forward_no_alpha(self, x):
        x_copy = x.to(self.dtype)
        return F.celu(x_copy, inplace=self.inplace), x_copy


class TestCelu(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(2, 4, 224, 224).astype(np.float32),)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("alpha", [None, 0.5, 2.])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
    @pytest.mark.parametrize("inplace", [skip_if_export(True), False])
    def test_celu(self, alpha, dtype, inplace, ie_device, precision, ir_version):
        kwargs = {}
        if dtype == torch.float16:
            kwargs["custom_eps"] = 1e-2
        self._test(aten_celu(alpha, dtype, inplace), None,
                   "aten::celu_" if inplace else "aten::celu", ie_device, precision, ir_version, **kwargs)
