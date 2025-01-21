# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestIndexFill(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_tensor,)

    def create_model(self, dim, index, values):
        class aten_index_fill_(torch.nn.Module):
            def __init__(self, dim, index, values):
                super().__init__()
                self.dim = dim
                self.index = index
                self.values = values

            def forward(self, input_tensor): 
                input_tensor.index_fill_(self.dim, self.index, self.values)
                return input_tensor

        ref_net = None

        return aten_index_fill_(dim, index, values), ref_net, "aten::index_fill_"

    @pytest.mark.parametrize(
        "input_data",
        (
            {
                "input_shape": [10],
                "dim": 0, 
                "input_value": 5.6,
                "index": [5, 6, 7]
            },
            {
                "input_shape": [3, 3],
                "dim": 0, 
                "input_value": 10.1,
                "index": [1, 0]
            },
            {
                "input_shape": [4, 3, 5],
                "dim": 1, 
                "input_value": 1234.5,
                "index": [2, 0]
            },
            {
                "input_shape": [5, 6, 7, 8],
                "dim": -2, 
                "input_value": 0.1234,
                "index": [6, 4, 2, 0]
            },
            {
                "input_shape": [5, 6, 7, 8],
                "dim": -3, 
                "input_value": -4321234.5678765,
                "index": [5, 4, 3, 1]
            },
            {
                "input_shape": [5, 6, 7, 8],
                "dim": 3, 
                "input_value": -1234.54321,
                "index": [6, 4, 7, 2, 1]
            },
        ),
    )
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_index_fill_single_index(self, ie_device, precision, ir_version, input_data):
        self.input_tensor = np.random.randn(*input_data["input_shape"]).astype(np.float32)
        values = torch.tensor(np.float32(input_data["input_value"]))
        dim = input_data["dim"]
        shape = self.input_tensor.shape
        max_idx = shape[dim]
        n_select = np.random.randint(1, max_idx + 1)
        index = torch.from_numpy(np.random.choice(np.arange(0, max_idx), n_select, replace=False)).to(torch.long)
        self._test(*self.create_model(dim, index, values), ie_device, precision, ir_version)