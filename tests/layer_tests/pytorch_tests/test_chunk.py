# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest

class aten_chunk_2(torch.nn.Module):
    def __init__(self, dim) -> None:
        torch.nn.Module.__init__(self)
        self.dim = dim

    def forward(self, input_tensor):
        a,b = torch.chunk(input_tensor, 
            chunks = 2,
            dim = self.dim
        )
        return a,b

class aten_chunk_3(torch.nn.Module):
    def __init__(self, dim) -> None:
        torch.nn.Module.__init__(self)
        self.dim = dim

    def forward(self, input_tensor):
        a,b,c = torch.chunk(input_tensor, 
            chunks = 3,
            dim = self.dim
        )
        return a,b,c

class aten_chunk_4(torch.nn.Module):
    def __init__(self, dim) -> None:
        torch.nn.Module.__init__(self)
        self.dim = dim

    def forward(self, input_tensor):
        a,b,c,d = torch.chunk(input_tensor, 
            chunks = 4,
            dim = self.dim
        )
        return a,b,c,d

class aten_chunk_getitem(torch.nn.Module):
    def __init__(self, chunks, dim, idx) -> None:
        torch.nn.Module.__init__(self)
        self.chunks = chunks
        self.dim = dim
        self.idx = idx

    def forward(self, input_tensor):
        return torch.chunk(input_tensor, 
            chunks = self.chunks,
            dim = self.dim
        )[self.idx]

class TestChunk(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.rand(*self.input_shape),)

    @pytest.mark.parametrize("input_shape", [
        (4, 4),
        (5, 9, 7),
        (10, 13, 11),
        (8, 7, 6, 5, 4),
    ])
    @pytest.mark.parametrize("chunks", [
        # Does not work for 1 - no list_unpack present in the graph
        # 1, 
        2,
        3,
        4
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_chunk(self, input_shape, chunks, ie_device, precision, ir_version):
        self.input_shape = input_shape
        
        for dim, dim_shape in enumerate(input_shape):
            chunk_size = dim_shape // chunks
            chunk_size += 1 if dim_shape % chunks > 0 else 0

            output_chunks = dim_shape // chunk_size
            output_chunks += 1 if dim_shape % chunk_size > 0 else 0
            
            if output_chunks == 2:
                cls = aten_chunk_2
            elif output_chunks == 3:
                cls = aten_chunk_3
            elif output_chunks == 4:
                cls = aten_chunk_4

            self._test(cls(dim), None, "aten::chunk", 
                    ie_device, precision, ir_version, dynamic_shapes = False, freeze_model=True, trace_model=True)
    
    @pytest.mark.parametrize("input_shape", [
        (4, 4),
        (10, 13, 11),
        (8, 7, 6, 5, 4),
    ])
    @pytest.mark.parametrize("chunks", [
        2,
        3,
        4
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_chunk_getitem(self, input_shape, chunks, ie_device, precision, ir_version):
        self.input_shape = input_shape
        for dim, dim_shape in enumerate(input_shape):

            chunk_size = dim_shape // chunks
            chunk_size += 1 if dim_shape % chunks > 0 else 0

            output_chunks = dim_shape // chunk_size
            output_chunks += 1 if dim_shape % chunk_size > 0 else 0

            for idx in [0, 1, output_chunks - 1]:
                self._test(aten_chunk_getitem(chunks, dim, idx), None, "aten::chunk", 
                            ie_device, precision, ir_version)


class aten_chunk_loop_getitem(torch.nn.Module):
    def __init__(self, num_chunks) -> None:
        torch.nn.Module.__init__(self)
        self.num_chunks = num_chunks

    def forward(self, input_tensor):
        chunks = torch.chunk(torch.arange(
            input_tensor.shape[0]), self.num_chunks)

        for inds in chunks:
            input_tensor[inds] *= 10
        return input_tensor


class TestChunkLoopGetitem(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.rand(*self.input_shape),)

    @pytest.mark.parametrize("input_shape", [
        (4, 4),
        (5, 9, 7),
        (10, 13, 11),
        (8, 7, 6, 5, 4),
    ])
    @pytest.mark.parametrize("chunks", [
        2,
        3,
        4
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_chunk_loop_getitem(self, input_shape, chunks, ie_device, precision, ir_version):
        self.input_shape = input_shape

        self._test(aten_chunk_loop_getitem(chunks), None, ["aten::chunk", "prim::Loop", "aten::__getitem__"],
                   ie_device, precision, ir_version)
