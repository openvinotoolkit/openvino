# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest


class TestNdtr(PytorchLayerTest):
    def _prepare_input(self, input_shape=(2, 2), dtype=np.float32):
        if np.issubdtype(dtype, np.integer):
            return (self.random.randint(-10, 10, size=input_shape, dtype=dtype),)
        return (self.random.uniform(-5.0, 5.0, size=input_shape, dtype=dtype),)

    def create_model(self, op_type="special_ndtr"):
        class aten_special_ndtr(torch.nn.Module):
            def forward(self, x):
                return torch.special.ndtr(x)

        class aten_ndtr(torch.nn.Module):
            def forward(self, x):
                return torch.ndtr(x)

        if op_type == "special_ndtr":
            return aten_special_ndtr(), "aten::special_ndtr"
        return aten_ndtr(), "aten::ndtr"

    @pytest.mark.parametrize("op_type", ["special_ndtr", "ndtr"])
    @pytest.mark.parametrize("input_shape", [(3, 3), (2, 3, 4)])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_ndtr(self, op_type, input_shape, dtype, ie_device, precision, ir_version):
        self._test(
            *self.create_model(op_type),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"input_shape": input_shape, "dtype": dtype},
        )
