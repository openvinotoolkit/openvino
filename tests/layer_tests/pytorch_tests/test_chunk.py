# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest

class aten_chunk(torch.nn.Module):
    def __init__(self, dim, chunks) -> None:
        torch.nn.Module.__init__(self)
        self.chunks = chunks
        self.dim = dim

    def forward(self, input_tensor):
        output =  torch.chunk(input_tensor, 
            chunks = self.chunks,
            dim = self.dim
        )
        print(f"Torch output len: {len(output)}")
        return output

class TestChunk(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_tensor,)

    @pytest.mark.parametrize("input_tensor", [
        # np.random.rand(1, 4),
        np.random.rand(4, 4),
        # np.random.rand(4, 4, 4),
        # np.random.rand(10, 10, 10),
        # np.random.rand(8, 8, 8, 8, 8)
    ])
    @pytest.mark.parametrize("chunks", [
        # 1,
        2,
        # 7,
        # 10
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_argsort(self, input_tensor, chunks, ie_device, precision, ir_version):
        self.input_tensor = input_tensor
        dims = len(input_tensor.shape)
        for dim in range(0, 1):
            self._test(aten_chunk(dim, chunks), None, ["aten::chunk", "prim::ListConstruct"], 
                    ie_device, precision, ir_version)