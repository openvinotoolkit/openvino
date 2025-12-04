# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest

class TestReplace(PytorchLayerTest):
    def _prepare_input(self, input_shape, input_dtype, target_val, replacement_val):
        return (
            np.random.randint(0, 10, input_shape).astype(input_dtype),
            np.array(target_val, dtype=input_dtype),
            np.array(replacement_val, dtype=input_dtype)
        )

    def create_model(self):
        class aten_replace_model(torch.nn.Module):
            def forward(self, x, target, replacement):
                return torch.where(x == target, replacement, x)

        ref_net = None

        return aten_replace_model(), ref_net, "aten::where"

    @pytest.mark.parametrize("input_shape", [
        [10],
        [5, 10],
        [2, 3, 4, 5]
    ])
    @pytest.mark.parametrize("input_dtype", [
        np.float32,
        np.int32,
    ])
    @pytest.mark.parametrize("target_val, replacement_val", [
        (5, 42),
        (0, -1)
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_replace(self, ie_device, precision, ir_version, input_shape, input_dtype, target_val, replacement_val):
        self._test(*self.create_model(),
                   ie_device,
                   precision,
                   ir_version,
                   kwargs_to_prepare_input={
                       "input_shape": input_shape,
                       "input_dtype": input_dtype,
                       "target_val": target_val,
                       "replacement_val": replacement_val
                   })
