# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


@pytest.mark.parametrize('input_tensor', (np.array([2, 0, 1, -2]),))
class TestIsFinite(PytorchLayerTest):

    def _prepare_input(self):
        input_tensor = self.input_tensor
        return (input_tensor,)

    def create_model(self):
        class aten_isfinite(torch.nn.Module):

            def forward(self, input_tensor):
                return torch.isfinite(torch.pow(input_tensor.to(torch.float32), 1000))

        return aten_isfinite(), None, "aten::isfinite"

    @pytest.mark.precommit_fx_backend
    def test_isfinite(self, ie_device, precision, ir_version, input_tensor):
        self.input_tensor = input_tensor
        self._test(*self.create_model(), ie_device, precision, ir_version)
