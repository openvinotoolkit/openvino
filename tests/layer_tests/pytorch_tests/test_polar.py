# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import openvino as ov
from pytorch_layer_test_class import PytorchLayerTest

class TestPolar(PytorchLayerTest):
    def _prepare_input(self, input_shape=(1, 1000), dtype=np.float32):
        return (
            np.random.uniform(0, 10, input_shape).astype(dtype),
            np.random.uniform(-np.pi, np.pi, input_shape).astype(dtype)
        )

    def create_model(self):
        class PolarModel(torch.nn.Module):
            def forward(self, abs, angle):
                complex_tensor = torch.polar(abs, angle)
                return torch.view_as_real(complex_tensor)
        ref_net = None
        return PolarModel(), None, "aten::polar"

    @pytest.mark.parametrize("input_case", [
        (1, 1000),
        (2, 500),
        (5, 200),
        (10, 100),
    ])
    @pytest.mark.parametrize("dtype", [
        np.float32,
        np.float64
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_polar(self, input_case, dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"input_shape": input_case, "dtype": dtype},
                   trace_model=True, use_convert_model=True, dynamic_shapes=False)
