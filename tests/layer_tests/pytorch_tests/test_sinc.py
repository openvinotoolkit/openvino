# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest


class TestSinc(PytorchLayerTest):
    def _prepare_input(self, input_shape=(3, 3), dtype=np.float32):
        inp = self.random.uniform(-5.0, 5.0, size=input_shape).astype(dtype)
        # Ensure at least one element is exactly 0 to test the x == 0 branch
        if inp.size > 0:
            inp.flat[0] = 0
        return (inp,)

    def create_model(self):
        class aten_sinc(torch.nn.Module):
            def forward(self, x):
                return torch.sinc(x)

        return aten_sinc(), "aten::sinc"

    @pytest.mark.parametrize("input_shape", [(2, 3), (3, 3, 2)])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.float16, np.int32])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_sinc(self, input_shape, dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"input_shape": input_shape, "dtype": dtype})
