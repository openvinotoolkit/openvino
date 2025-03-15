# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import torch
from pytorch_layer_test_class import PytorchLayerTest

class TestTake(PytorchLayerTest):
    def _prepare_input(self, input_shape, indices_shape, max_val):
        input_tensor = np.random.randn(*input_shape).astype(np.float32)
        indices = np.random.randint(-max_val, max_val, indices_shape).astype(np.int64)
        return (input_tensor, indices)

    def create_model(self):
        class aten_take(torch.nn.Module):
            def forward(self, x, indices):
                return torch.take(x, indices)

        ref_net = None
        return aten_take(), ref_net, "aten::take"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("input_shape", [(10,), (3, 4), (2, 3, 4), (100,)])
    @pytest.mark.parametrize("indices_shape", [(5,), (2, 2), (3, 2), (50,)])
    def test_take(self, input_shape, indices_shape, ie_device, precision, ir_version):
        max_val = np.prod(input_shape)
        self._test(*self.create_model(), ie_device, precision, ir_version, 
                   kwargs_to_prepare_input={
                       "input_shape": input_shape,
                       "indices_shape": indices_shape,
                       "max_val": max_val
                   })
