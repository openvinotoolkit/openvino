# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


@pytest.mark.parametrize('input_tensor', (torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')])))
class TestIsFinite(PytorchLayerTest):

    def _prepare_input(self):
        input_tensor = self.input_tensor
        return (input_tensor,)

    def create_model(self):
        class aten_isfinite(torch.nn.Module):

            def forward(self, input_tensor):
                return torch.isfinite(input_tensor)

        ref_net = None

        return aten_isfinite(), ref_net, "aten::isfinite"

    @pytest.mark.precommit_fx_backend
    def test_isfinite(self, ie_device, precision, ir_version, input_tensor):
        self.input_tensor = input_tensor
        self._test(*self.create_model(), ie_device, precision, ir_version)
