# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0i

import pytest
import torch
import math
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest

class TestAtan2(PytorchLayerTest):
    def _prepare_input(self, x, y, dtype=None):
        inputs_data = {}
        inputs_data['y'] = np.array(y).astype(dtype) - np.array(y).astype(dtype)
        inputs_data['x'] = np.array(x).astype(dtype) - np.array(x).astype(dtype)
        return inputs_data

    def create_model(self, input_type):
        class aten_atan2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.input_type = input_type

            def forward(self, input_tensor, other_tensor):
                return torch.atan2(input_tensor.to(self.input_type), other_tensor.to(self.input_type))

        ref_net = None

        return aten_atan2(), ref_net, "aten::atan2"

    @pytest.mark.parametrize(("input_type"), [
        (torch.float16),
        (torch.int32),
        (torch.float64),
        (torch.float32),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_atan2(self, input_type, ie_device, precision, ir_version):
        self._test(*self.create_model(input_type), ie_device, precision, ir_version, use_convert_model=True)
