# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestHardtanh(PytorchLayerTest):
    def _prepare_input(self):
        return (np.round(np.array(5.00 * np.random.rand(10, 10) - 2.50, dtype=np.float32), 4),)

    def create_model(self, min_val, max_val, inplace):
        import torch
        import torch.nn.functional as F

        class aten_hardtanh(torch.nn.Module):
            def __init__(self, min_val, max_val, inplace):
                super(aten_hardtanh, self).__init__()
                self.min_val = min_val
                self.max_val = max_val
                self.inplace = inplace

            def forward(self, x):
                return F.hardtanh(x, min_val=self.min_val, max_val=self.max_val, inplace=self.inplace)

        ref_net = None

        return aten_hardtanh(min_val, max_val, inplace), ref_net, "aten::hardtanh"

    @pytest.mark.parametrize(("min_val", "max_val"), [[-1.0,1.0], [0, 1.0], [-2.0, 2.0]])
    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    def test_hardtanh(self, min_val, max_val, inplace, ie_device, precision, ir_version):
        self._test(*self.create_model(min_val, max_val, inplace), ie_device, precision, ir_version)
