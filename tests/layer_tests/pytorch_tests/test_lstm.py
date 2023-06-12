# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest

class aten_lstm(torch.nn.Module):
    def __init__(self, hx, params, num_layers, has_biases, bidirectional) -> None:
        torch.nn.Module.__init__(self)
        self.hx = hx
        self.params = params, 
        self.num_layers = layers, 
        self.has_biases = has_biases, 
        self.bidirectional = bidirectional

    def forward(self, input_tensor):
        return torch._VF.lstm(input_tensor, 
                              self.hx,
                              self.params,
                              self.has_biases,
                              self.num_layers,
                              False,
                              False,
                              self.bidirectional)

class TestLSTM(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_tensor,)

    @pytest.mark.parametrize("input_shape", [
        np.random.randint(0, 2, (2))
    ])
    @pytest.mark.parametrize("num_layers", [
        np.random.randint(1, 4)
    ])
    @pytest.mark.parametrize("has_biases", [
        True, False
    ])
    @pytest.mark.parametrize("bidirectional", [
        True, False
    ])
    @pytest.mark.parametrize("projection", [
        True, False
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_all_noparams(self, input_shape, num_layers, has_biases, bidirectional, projection, ie_device, precision, ir_version):
        self.input_tensor = np.random.rand(*input_shape)

        hx = [np.random.rand(*input_shape), np.random.rand(*input_shape)]
        params = []
        for i in range(num_layers):
            for j in range(int(bidirectional) + 1):
                m = 5 if projection else 4
                params.append(np.random.rand(input_shape[0] * m, input_shape[1]))
                params.append(np.random.rand(input_shape[0] * m, input_shape[1]))
                if has_biases:
                    params.append(np.random.rand(input_shape[0] * m, 1))
                    params.append(np.random.rand(input_shape[0] * m, 1))
        self._test(aten_lstm(hx, params, num_layers, has_biases, bidirectional), None, "aten::lstm", 
                ie_device, precision, ir_version, trace_model=True, freeze_model=False)
