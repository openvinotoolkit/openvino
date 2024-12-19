# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from packaging import version

from pytorch_layer_test_class import PytorchLayerTest, skip_if_export


class TestLerp(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(2, 5, 3, 4).astype(np.float32), self.input_rhs)

    def create_model(self, weight, op_type):
        class aten_lerp(torch.nn.Module):
            def __init__(self, weight, op) -> None:
                super().__init__()
                self.weight = weight
                self.forward = self.forward1 if op == "lerp" else self.forward2

            def forward1(self, lhs, rhs):
                return torch.lerp(lhs, rhs, weight=self.weight)

            def forward2(self, lhs, rhs):
                return lhs.lerp_(rhs, weight=self.weight)

        return aten_lerp(weight, op_type), None, f"aten::{op_type}"

    @pytest.mark.parametrize("weight", (-0.5,
                                        0,
                                        0.5,
                                        1,
                                        2,
                                        skip_if_export([1, 5, 3, 4]))
                             )
    @pytest.mark.parametrize("input_shape_rhs", [[2, 5, 3, 4],
                                                 [1, 5, 3, 4],
                                                 [1]])
    @pytest.mark.parametrize("op_type", ["lerp", "lerp_"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_lerp(self, ie_device, precision, ir_version,
                  weight, input_shape_rhs, op_type):
        if (op_type == "lerp_" and PytorchLayerTest.use_torch_export() and
                version.parse(torch.__version__) < version.parse("2.5")):
            pytest.skip("Not supported in PyTorch versions earlier than 2.5.")
        self.input_rhs = np.random.randn(*input_shape_rhs).astype(np.float32)
        if isinstance(weight, list):
            weight = torch.rand(weight)
        self._test(
            *self.create_model(weight, op_type),
            ie_device,
            precision,
            ir_version,
            use_convert_model=True,
        )
