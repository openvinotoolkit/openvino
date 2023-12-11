# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestSquareTypes(PytorchLayerTest):

    def _prepare_input(self):
        return (torch.randn(self.lhs_shape).to(self.lhs_type).numpy(),)

    def create_model(self, lhs_type, lhs_shape):

        class aten_square(torch.nn.Module):
            def __init__(self, lhs_type, lhs_shape):
                super().__init__()
                self.lhs_type = lhs_type

            def forward(self, lhs):
                return torch.square(lhs.to(self.lhs_type))

        return aten_square(lhs_type, lhs_shape), None, "aten::square"

    @pytest.mark.parametrize(("lhs_type"), [torch.int32, torch.int64, torch.float32])
    @pytest.mark.parametrize(("lhs_shape"), [[2, 3], [],])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_square_types(self, ie_device, precision, ir_version, lhs_type, lhs_shape):
        self.lhs_type = lhs_type
        self.lhs_shape = lhs_shape
        self._test(*self.create_model(lhs_type, lhs_shape),
                   ie_device, precision, ir_version)
