# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class aten_logsumexp(torch.nn.Module):
    def __init__(self, dim, keepdim) -> None:
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, input_tensor):
        return torch.logsumexp(input_tensor, dim=self.dim, keepdim=self.keepdim)


class TestLogsumexp(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(2, 5, 9, 7),)

    @pytest.mark.parametrize("dim", [
        0, 1, 2, 3, -1, -2, -3, -4
    ])
    @pytest.mark.parametrize("keepdim", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_logsumexp(self, dim, keepdim, ie_device, precision, ir_version):
        self._test(aten_logsumexp(dim, keepdim), None, "aten::logsumexp",
                   ie_device, precision, ir_version)
