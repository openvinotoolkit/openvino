# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest

class aten_lstm(torch.nn.Module):
    def __init__(self, hx, params, has_bias, num_layers, bidirectional, batch_first) -> None:
        torch.nn.Module.__init__(self)
        self.hx = hx
        self.params = params, 
        self.num_layers = num_layers, 
        self.has_bias = has_bias, 
        self.bidirectional = bidirectional
        self.batch_first = batch_first

    def forward(self, input_tensor):
        return torch._VF.lstm(input_tensor, 
                              self.hx,
                              self.params,
                              self.has_bias,
                              self.num_layers,
                              False,
                              False,
                              self.bidirectional,
                              self.batch_first)

class TestLSTM(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_tensor,)

    @pytest.mark.parametrize("input_shape", [
        [4, 2, 2],
        [10, 3, 3]
    ])
    @pytest.mark.parametrize("num_layers", [
        1,
        2
    ])
    @pytest.mark.parametrize("has_bias", [
        True, False
    ])
    @pytest.mark.parametrize("bidirectional", [
        True, False
    ])
    @pytest.mark.parametrize("batch_first", [
        True, False
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_lstm(self, input_shape, num_layers, has_bias, bidirectional, batch_first, ie_device, precision, ir_version):
        if not batch_first:
            input_shape = input_shape[1:].append(input_shape[0])
        self.input_tensor = np.random.rand(*input_shape)
        hx = [np.random.rand(*input_shape), np.random.rand(*input_shape)]
        params = []
        for i in range(num_layers):
            for j in range(int(bidirectional) + 1):
                m = 4
                params.append(np.random.rand(input_shape[0] * m, input_shape[1]))
                params.append(np.random.rand(input_shape[0] * m, input_shape[1]))
                if has_bias:
                    params.append(np.random.rand(input_shape[0] * m, 1))
                    params.append(np.random.rand(input_shape[0] * m, 1))
        self._test(aten_lstm(hx, params, has_bias, num_layers, bidirectional, batch_first), None, "aten::lstm", 
                ie_device, precision, ir_version)
