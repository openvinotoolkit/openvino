# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest

# Models that generate aten ops directly
class CallFunctionReLUModel(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.relu(x)

class CallFunctionAbsModel(torch.nn.Module):
    def forward(self, x):
        return torch.abs(x)

# Custom function that becomes aten::relu
@torch.jit.script
def custom_activation(x):
    return torch.relu(x)

class CallFunctionCustomModel(torch.nn.Module):
    def forward(self, x):
        return custom_activation(x)


class TestCallFunction(PytorchLayerTest):
    def _prepare_input(self, dtype=np.float32):
        # Default method for generating random inputs
        return (np.random.randn(2, 3, 4, 5).astype(dtype),)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_relu(self, ie_device, precision, ir_version):
        model = CallFunctionReLUModel()
        # ref_net=None tells the runner to use the _prepare_input method
        self._test(model, None, "aten::relu", ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_abs(self, ie_device, precision, ir_version):
        model = CallFunctionAbsModel()
        self._test(model, None, "aten::abs", ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("dtype", [np.float32, np.float16])
    def test_relu_types(self, ie_device, precision, ir_version, dtype):
        model = CallFunctionReLUModel()
        # The runner will call _prepare_input(dtype=dtype)
        self._test(model, None, "aten::relu", ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"dtype": dtype})

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_custom(self, ie_device, precision, ir_version):
        model = CallFunctionCustomModel()
        self._test(model, None, "aten::relu", ie_device, precision, ir_version)

    @pytest.mark.nightly
    def test_relu_zeros(self, ie_device, precision, ir_version):
        model = CallFunctionReLUModel()
        # Provide inputs directly as the second argument (ref_net)
        inputs = (np.zeros((2, 3), dtype=np.float32),)
        self._test(model, inputs, "aten::relu", ie_device, precision, ir_version)

    @pytest.mark.nightly
    def test_relu_ones(self, ie_device, precision, ir_version):
        model = CallFunctionReLUModel()
        inputs = (np.ones((2, 3), dtype=np.float32),)
        self._test(model, inputs, "aten::relu", ie_device, precision, ir_version)

    @pytest.mark.nightly
    def test_abs_negative(self, ie_device, precision, ir_version):
        model = CallFunctionAbsModel()
        inputs = (np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype=np.float32),)
        self._test(model, inputs, "aten::abs", ie_device, precision, ir_version)