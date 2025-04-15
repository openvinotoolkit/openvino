# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest


class TestReverse(PytorchLayerTest):
    def _prepare_input(self, dims):
        # Input tensor of shape [2, 3, 4] with float32 values
        return (np.random.randn(2, 3, 4).astype(np.float32),)

    def create_model(self, dims):
        class ReverseModel(torch.nn.Module):
            def __init__(self, dims):
                super().__init__()
                self.dims = dims

            def forward(self, x):
                # Using torch.flip which internally emits aten::flip
                return torch.flip(x, self.dims)

        # Update: using "aten::flip" instead of "aten::reverse"
        return ReverseModel(dims), None, "aten::flip"

    @pytest.mark.parametrize("dims", [
        [0], [1], [2], [1, 2], [-1], [-2], [0, 2], [0, 1, 2]
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_reverse(self, dims, ie_device, precision, ir_version):
        print(f"\n Running test for dims={dims}")
        self._test(*self.create_model(dims),
                   ie_device=ie_device,
                   precision=precision,
                   ir_version=ir_version,
                   kwargs_to_prepare_input={"dims": dims})