# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestSquareTypes(PytorchLayerTest):

    def _prepare_input(self):
        return (torch.randn(self.shape).to(self.type).numpy(),)

    def create_model(self, type):

        class aten_square(torch.nn.Module):
            def __init__(self, type):
                super().__init__()
                self.type = type

            def forward(self, lhs):
                return torch.square(lhs.to(self.type))

        return aten_square(type), None, "aten::square"

    @pytest.mark.parametrize(("type"), [torch.int32, torch.int64, torch.float32])
    @pytest.mark.parametrize(("shape"), [[2, 3], [],])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_square_types(self, ie_device, precision, ir_version, type, shape):
        if ie_device == "GPU" and type != torch.float32:
            pytest.xfail(reason="square is not supported on GPU for integer types")
        self.type = type
        self.shape = shape
        self._test(*self.create_model(type),
                   ie_device, precision, ir_version)
