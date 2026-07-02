# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
import openvino as ov
from typing import Dict, Tuple
from pytorch_layer_test_class import PytorchLayerTest

# Need to test for TorchScript and FX models with different input types for all scenarios

# Scenario 1: Dict input
# aten::values(Dict(a) self) -> Dict[a]
# Scenario 2: Tensor input
# aten::values(Tensor(a) self) -> Tensor(a)

def make_dict(input_tensor: torch.Tensor):
    return {0: input_tensor, 1: input_tensor + input_tensor, 2: 2 * input_tensor}

class aten_values_dict_input(torch.nn.Module):
    def forward(self, input_tensor: torch.Tensor):
        # x : Dict[int, torch.Tensor] = {idx:tensor for idx, tensor in enumerate(input_tuple)}
        # x : Dict[int, torch.Tensor] = {0: input_tensor, 1: input_tensor + input_tensor, 2: 2 * input_tensor}
        x = make_dict(input_tensor)
        return x.values()

class TestDictInput(PytorchLayerTest):
    def _generate_tensor(self):
        return self.random.randn(2, 5, 3, 4)
    def _prepare_input(self):
        return (self._generate_tensor(),)
    def create_model(self):
        return aten_values_dict_input(), "aten::values"
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_dict_input(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision,
                   ir_version, use_convert_model=True)