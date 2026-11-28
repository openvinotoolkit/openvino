# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest


class TestDeg2Rad(PytorchLayerTest):
    def _prepare_input(self, input_shape=(2, 2), dtype=np.float32):
        return (self.random.uniform(-360.0, 360.0, size=input_shape, dtype=dtype),)

    def create_model(self):
        class aten_deg2rad(torch.nn.Module):
            def forward(self, x):
                return torch.deg2rad(x)
        return aten_deg2rad(), "aten::deg2rad"

    @pytest.mark.parametrize("input_shape", [(3, 3)])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_deg2rad(self, input_shape, dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"input_shape": input_shape, "dtype": dtype})
