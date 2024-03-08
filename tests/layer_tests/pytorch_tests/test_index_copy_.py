# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestIndexCopy_SingleIndex(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_tensor, self.values)

    def create_model(self, dim, index):
        class aten_index_copy_(torch.nn.Module):
            def __init__(self, dim, index):
                super().__init__()
                self.dim = dim
                self.index = index

            def forward(self, input_tensor, values):
                input_tensor.index_copy_(self.dim, self.index, values)
                return input_tensor

        ref_net = None

        return aten_index_copy_(dim, index), ref_net, "aten::index_copy_"

    @pytest.mark.parametrize(
        "input_data",
        (
            # {
            #     "input_shape": [5],
            #     "dim": 0, 
            #     "values": np.array([1, 2, 3]).astype(np.float32), 
            #     "index": torch.tensor([0, 2, 1], dtype=torch.long)
            # },
            # {
            #     "input_shape": [3, 3],
            #     "dim": 0, 
            #     "values": np.array([[10, 11, 12], [-1, -2, -3]]).astype(np.float32),
            #     "index": torch.tensor([2, 0], dtype=torch.long)
            # },
            {
                "input_shape": [3, 3],
                "dim": 0, 
                "values": np.array([[7, 8], [-1, -2], [0, 5]]).astype(np.float32),
                "index": torch.tensor([1, 0], dtype=torch.long)
            },
        ),
    )
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_index_copy_single_index(self, ie_device, precision, ir_version, input_data):
        self.input_tensor = np.random.randn(*input_data["input_shape"]).astype(np.float32)
        self.values = input_data["values"]
        index = input_data["index"]
        dim = input_data["dim"]
        self._test(*self.create_model(dim, index), ie_device, precision, ir_version)
