# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestHardswish(PytorchLayerTest):
    def _prepare_input(self):
        return (np.round(np.array(5.00 * np.random.rand(10, 10) - 2.50, dtype=np.float32), 4),)

    def create_model(self, inplace):
        import torch
        import torch.nn.functional as F

        class aten_hardswish(torch.nn.Module):
            def __init__(self, inplace):
                super(aten_hardswish, self).__init__()
                self.inplace = inplace

            def forward(self, x):
                return F.hardswish(x, inplace=self.inplace)

        ref_net = None

        return aten_hardswish(inplace), ref_net, "aten::hardswish"

    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    def test_hardswish(self, inplace, ie_device, precision, ir_version):
        self._test(*self.create_model(inplace), ie_device, precision, ir_version)
