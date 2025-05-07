# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestIndexCopy(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_tensor, self.values)

    def create_model(self, dim, index, inplace):
        class aten_index_copy_(torch.nn.Module):
            def __init__(self, dim, index, inplace):
                super().__init__()
                self.dim = dim
                self.index = index
                self.inplace = inplace

            def forward(self, input_tensor, values):
                if not self.inplace:
                    return input_tensor.index_copy(self.dim, self.index, values)
                input_tensor.index_copy_(self.dim, self.index, values)
                return input_tensor

        ref_net = None
        op_name = "aten::index_copy_" if inplace else "aten::index_copy"
        return aten_index_copy_(dim, index, inplace), ref_net, op_name

    @pytest.mark.parametrize(
        "input_data",
        (
            {
                "input_shape": [1],
                "dim": 0, 
                "values_shape": [1],
                "index": torch.tensor([0], dtype=torch.long)
            },
            {
                "input_shape": [10],
                "dim": 0, 
                "values_shape": [5],
                "index": torch.tensor([2, 3, 6, 7, 1], dtype=torch.long)
            },
            {
                "input_shape": [3, 3],
                "dim": 0, 
                "values_shape": [2, 3],
                "index": torch.tensor([2, 0], dtype=torch.long)
            },
            {
                "input_shape": [4, 3, 5],
                "dim": 1, 
                "values_shape": [4, 2, 5],
                "index": torch.tensor([1, 0], dtype=torch.long)
            },
            {
                "input_shape": [5, 6, 7, 8],
                "dim": -2, 
                "values_shape": [5, 6, 4, 8],
                "index": torch.tensor([5, 0, 6, 3], dtype=torch.long)
            },
            {
                "input_shape": [5, 6, 7, 8],
                "dim": -3, 
                "values_shape": [5, 3, 7, 8],
                "index": torch.tensor([2, 0, 1], dtype=torch.long)
            },
            {
                "input_shape": [5, 6, 7, 8],
                "dim": 3, 
                "values_shape": [5, 6, 7, 5],
                "index": torch.tensor([2, 6, 0, 4, 1], dtype=torch.long)
            },
        ),
    )
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize('inplace', [True, False])
    def test_index_copy_single_index(self, inplace, ie_device, precision, ir_version, input_data):
        self.input_tensor = np.random.randn(*input_data["input_shape"]).astype(np.float32)
        self.values = np.random.randn(*input_data["values_shape"]).astype(np.float32)
        index = input_data["index"]
        dim = input_data["dim"]
        self._test(*self.create_model(dim, index, inplace), ie_device, precision, ir_version)
