# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class aten_lstm(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, has_bias, bidirectional, batch_first):
        torch.nn.Module.__init__(self)
        self.lstm = torch.nn.LSTM(input_size,
                                  hidden_size,
                                  num_layers,
                                  has_bias,
                                  batch_first,
                                  bidirectional=bidirectional)

    def forward(self, input_tensor, h0, c0):
        return self.lstm(input_tensor, (h0, c0))


class TestLSTM(PytorchLayerTest):
    def _prepare_input(self):
        n = self.num_layers
        if self.bidirectional:
            n *= 2
        if self.batch_first:
            input = np.random.randn(3, 5, self.input_size).astype(np.float32)
        else:
            input = np.random.randn(5, 3, self.input_size).astype(np.float32)
        h0 = np.random.randn(n, 3, self.hidden_size).astype(np.float32)
        c0 = np.random.randn(n, 3, self.hidden_size).astype(np.float32)
        return (input, h0, c0)

    @pytest.mark.parametrize("input_size,hidden_size", [(10, 20),])
    @pytest.mark.parametrize("num_layers", [1, 2, 7])
    @pytest.mark.parametrize("has_bias", [True, False])
    @pytest.mark.parametrize("bidirectional", [True, False])
    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_lstm(self, input_size, hidden_size, num_layers, has_bias, bidirectional, batch_first, ie_device, precision, ir_version):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self._test(aten_lstm(input_size, hidden_size, num_layers, has_bias, bidirectional, batch_first), None, "aten::lstm",
                   ie_device, precision, ir_version, trace_model=True)
