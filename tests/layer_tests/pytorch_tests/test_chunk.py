# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class aten_chunk_2(torch.nn.Module):
    def __init__(self, dim, unsafe=False) -> None:
        torch.nn.Module.__init__(self)
        self.dim = dim
        self.chunk_op = torch.chunk
        if unsafe:
            self.chunk_op = torch._VF.unsafe_chunk

    def forward(self, input_tensor):
        a, b = self.chunk_op(input_tensor,
                             chunks=2,
                             dim=self.dim
                             )
        return a, b


class aten_chunk_3(torch.nn.Module):
    def __init__(self, dim, unsafe=False) -> None:
        torch.nn.Module.__init__(self)
        self.dim = dim
        self.chunk_op = torch.chunk
        if unsafe:
            self.chunk_op = torch._VF.unsafe_chunk

    def forward(self, input_tensor):
        a, b, c = self.chunk_op(input_tensor,
                                chunks=3,
                                dim=self.dim
                                )
        return a, b, c


class aten_chunk_4(torch.nn.Module):
    def __init__(self, dim, unsafe=False) -> None:
        torch.nn.Module.__init__(self)
        self.dim = dim
        self.chunk_op = torch.chunk
        if unsafe:
            self.chunk_op = torch._VF.unsafe_chunk

    def forward(self, input_tensor):
        a, b, c, d = self.chunk_op(input_tensor,
                                   chunks=4,
                                   dim=self.dim
                                   )
        return a, b, c, d


class aten_chunk_getitem(torch.nn.Module):
    def __init__(self, chunks, dim, idx, unsafe=False) -> None:
        torch.nn.Module.__init__(self)
        self.chunks = chunks
        self.dim = dim
        self.idx = idx
        self.chunk_op = torch.chunk
        if unsafe:
            self.chunk_op = torch._VF.unsafe_chunk

    def forward(self, input_tensor):
        return self.chunk_op(input_tensor,
                             chunks=self.chunks,
                             dim=self.dim
                             )[self.idx]


class TestChunk(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.rand(*self.input_shape),)

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
    @pytest.mark.parametrize("unsafe", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_chunk(self, input_shape, chunks, unsafe, ie_device, precision, ir_version):
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

            self._test(cls(dim, unsafe),
                       "aten::unsafe_chunk" if unsafe else "aten::chunk",
                       ie_device, precision, ir_version, dynamic_shapes=False,
                       freeze_model=True, trace_model=True)

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
    @pytest.mark.parametrize("unsafe", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_chunk_getitem(self, input_shape, chunks, unsafe, ie_device, precision, ir_version):
        self.input_shape = input_shape
        for dim, dim_shape in enumerate(input_shape):

            chunk_size = dim_shape // chunks
            chunk_size += 1 if dim_shape % chunks > 0 else 0

            output_chunks = dim_shape // chunk_size
            output_chunks += 1 if dim_shape % chunk_size > 0 else 0

            for idx in [0, 1, output_chunks - 1]:
                self._test(aten_chunk_getitem(chunks, dim, idx, unsafe),
                           "aten::unsafe_chunk" if unsafe else "aten::chunk",
                           ie_device, precision, ir_version)


class aten_chunk_loop_getitem(torch.nn.Module):
    def __init__(self, num_chunks, unsafe=False) -> None:
        torch.nn.Module.__init__(self)
        self.num_chunks = num_chunks
        self.chunk_op = torch.chunk
        if unsafe:
            self.chunk_op = torch._VF.unsafe_chunk

    def forward(self, input_tensor):
        x = torch.arange(input_tensor.shape[0])
        chunks = self.chunk_op(x, self.num_chunks)

        for inds in chunks:
            input_tensor[inds] *= 10
        return input_tensor


class TestChunkLoopGetitem(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.rand(*self.input_shape),)

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
    @pytest.mark.parametrize("unsafe", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_chunk_loop_getitem(self, input_shape, chunks, unsafe, ie_device, precision, ir_version):
        self.input_shape = input_shape

        chunk_op = "aten::unsafe_chunk" if unsafe else "aten::chunk"
        self._test(aten_chunk_loop_getitem(chunks, unsafe),
                   [chunk_op, "prim::Loop", "aten::__getitem__"],
                   ie_device, precision, ir_version)


class aten_chunk_neg_dim_3(torch.nn.Module):
    """Model for testing chunk with a negative dim argument (as used in SAM3 qkv.chunk(3, dim=-1))."""

    def __init__(self, dim) -> None:
        torch.nn.Module.__init__(self)
        self.dim = dim

    def forward(self, x):
        a, b, c = torch.chunk(x, chunks=3, dim=self.dim)
        return a, b, c


class TestChunkNegativeDim(PytorchLayerTest):
    """Tests that aten::chunk works correctly with negative dimension indices."""

    def _prepare_input(self):
        return (self.random.rand(*self.input_shape),)

    @pytest.mark.parametrize("input_shape,dim", [
        # 3D tensor; last dim has size 12 so chunks of 4
        ((4, 6, 12), -1),
        ((4, 6, 12), -3),
        # 4D tensor; various negative dims
        ((2, 4, 6, 9), -1),
        ((2, 4, 6, 9), -2),
        ((2, 4, 6, 9), -4),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_chunk_negative_dim(self, input_shape, dim, ie_device, precision, ir_version):
        self.input_shape = input_shape
        self._test(aten_chunk_neg_dim_3(dim),
                   "aten::chunk",
                   ie_device, precision, ir_version,
                   dynamic_shapes=False, freeze_model=True, trace_model=True)
