# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


@pytest.mark.parametrize('input_tensor', (np.array([1, 0, -1]),))
class TestIsInf(PytorchLayerTest):

    def _prepare_input(self):
        input_tensor = self.input_tensor
        return (input_tensor,)

    def create_model(self):
        class aten_isinf(torch.nn.Module):

            def forward(self, input_tensor):
                return torch.isinf(input_tensor * float("inf"))

        return aten_isinf(), None, "aten::isinf"

    @pytest.mark.precommit_fx_backend
    def test_isinf(self, ie_device, precision, ir_version, input_tensor):
        self.input_tensor = input_tensor
        self._test(*self.create_model(), ie_device, precision, ir_version)
