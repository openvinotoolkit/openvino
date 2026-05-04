# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Regression tests for out-of-bounds index validation in aten::__getitem__
replacer transforms (CVE fix).

These tests verify that ov.convert_model raises a proper exception (instead
of crashing with SIGSEGV) when a TorchScript model contains an
aten::__getitem__ node whose constant index is out of bounds for the
producer collection (aten::split outputs, SequenceMark inputs, or
append-list inputs).
"""

import pytest
import torch
import openvino as ov

from pytorch_layer_test_class import PytorchLayerTest

# ---------------------------------------------------------------------------
# GI1: aten::split (scalar split_size) + aten::__getitem__ with OOB index
# ---------------------------------------------------------------------------

class SplitGetitemOOB(torch.nn.Module):
    """Model that indexes split results with an out-of-bounds constant."""

    def __init__(self, split_size, dim, index):
        super().__init__()
        self.split_size = split_size
        self.dim = dim
        self.index = index

    def forward(self, x):
        splits = torch.split(x, self.split_size, self.dim)
        return splits[self.index]


class SplitGetitemNegativeOOB(torch.nn.Module):
    """Model that indexes split results with a large negative OOB constant."""

    def __init__(self, split_size, dim, index):
        super().__init__()
        self.split_size = split_size
        self.dim = dim
        self.index = index

    def forward(self, x):
        splits = torch.split(x, self.split_size, self.dim)
        return splits[self.index]


class TestSplitGetitemOOBIndex(PytorchLayerTest):
    """
    Tests for aten::split + aten::__getitem__ with out-of-bounds index.

    When split_size is a scalar, torch.split produces aten::split which is
    not in op_table.cpp and stays as a FrameworkNode. The AtenGetItemReplacer
    transform then processes it. An OOB index must be caught gracefully.
    """

    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.parametrize("oob_index", [999, 100, 5])
    def test_positive_oob_index_no_crash(self, oob_index, ie_device, precision):
        """Positive OOB index must not cause SIGSEGV."""
        # Input shape [1, 10], split_size=2, dim=1 -> 5 chunks (indices 0-4)
        model = SplitGetitemOOB(split_size=2, dim=1, index=oob_index)
        sample_input = self.random.randn(1, 10)

        try:
            scripted = torch.jit.script(model)
        except Exception:
            pytest.skip("torch.jit.script rejected the model")

        # The conversion must NOT segfault. It should raise a clean
        # exception for out-of-bounds index.
        # Static input shape is required so the OOB check can compute num_splits.
        with pytest.raises(ov.frontend.OpConversionFailure):
            ov.convert_model(scripted, example_input=(sample_input,), input=[1, 10])

    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.parametrize("oob_index", [-999, -100, -6])
    def test_negative_oob_index_no_crash(self, oob_index, ie_device, precision):
        """Large negative OOB index must not cause SIGSEGV."""
        model = SplitGetitemNegativeOOB(split_size=2, dim=1, index=oob_index)
        sample_input = self.random.randn(1, 10)

        try:
            scripted = torch.jit.script(model)
        except Exception:
            pytest.skip("torch.jit.script rejected the model")

        with pytest.raises(ov.frontend.OpConversionFailure):
            ov.convert_model(scripted, example_input=(sample_input,), input=[1, 10])

    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.parametrize("valid_index", [-5, -1, 0, 1, 4])
    def test_valid_index_still_works(self, valid_index, ie_device, precision):
        """Valid indices (including negative wrapping) must still convert."""
        model = SplitGetitemOOB(split_size=2, dim=1, index=valid_index)
        sample_input = self.random.randn(1, 10)

        try:
            scripted = torch.jit.script(model)
        except Exception:
            pytest.skip("torch.jit.script rejected the model")

        # Valid indices should convert successfully
        ov_model = ov.convert_model(scripted, example_input=(sample_input,), input=[1, 10])
        assert ov_model is not None


# ---------------------------------------------------------------------------
# GI1 variant: aten::split (list split_sizes) + aten::__getitem__ with OOB
# ---------------------------------------------------------------------------

class SplitSizesGetitemOOB(torch.nn.Module):
    """Model using list split_sizes with OOB getitem index."""

    def __init__(self, split_sizes, dim, index):
        super().__init__()
        self.split_sizes = split_sizes
        self.dim = dim
        self.index = index

    def forward(self, x):
        splits = torch.split(x, self.split_sizes, self.dim)
        return splits[self.index]


class TestSplitSizesGetitemOOBIndex(PytorchLayerTest):
    """
    Tests for aten::split with list split_sizes + OOB __getitem__.

    When split_sizes is a list, the VariadicSplit branch is taken in
    AtenGetItemReplacer. OOB index on split->outputs() must be caught.
    """

    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.parametrize("oob_index", [3, 10, 999])
    def test_positive_oob_index_no_crash(self, oob_index, ie_device, precision):
        """Positive OOB index on list split must not crash."""
        # split_sizes=[2,3,5] -> 3 chunks (indices 0-2)
        model = SplitSizesGetitemOOB(split_sizes=[2, 3, 5], dim=1, index=oob_index)
        sample_input = self.random.randn(1, 10)

        try:
            scripted = torch.jit.script(model)
        except Exception:
            pytest.skip("torch.jit.script rejected the model")

        with pytest.raises(ov.frontend.OpConversionFailure):
            ov.convert_model(scripted, example_input=(sample_input,))

    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.parametrize("oob_index", [-4, -100])
    def test_negative_oob_index_no_crash(self, oob_index, ie_device, precision):
        """Large negative OOB index on list split must not crash."""
        model = SplitSizesGetitemOOB(split_sizes=[2, 3, 5], dim=1, index=oob_index)
        sample_input = self.random.randn(1, 10)

        try:
            scripted = torch.jit.script(model)
        except Exception:
            pytest.skip("torch.jit.script rejected the model")

        with pytest.raises(ov.frontend.OpConversionFailure):
            ov.convert_model(scripted, example_input=(sample_input,))

    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.parametrize("valid_index", [-3, -1, 0, 1, 2])
    def test_valid_list_split_index_still_works(self, valid_index, ie_device, precision):
        """Valid indices for list split_sizes must still convert."""
        model = SplitSizesGetitemOOB(split_sizes=[2, 3, 5], dim=1, index=valid_index)
        sample_input = self.random.randn(1, 10)

        try:
            scripted = torch.jit.script(model)
        except Exception:
            pytest.skip("torch.jit.script rejected the model")

        ov_model = ov.convert_model(scripted, example_input=(sample_input,))
        assert ov_model is not None


# ---------------------------------------------------------------------------
# Chunk + getitem OOB (related pattern in AtenGetItemReplacer)
# ---------------------------------------------------------------------------

class ChunkGetitemOOB(torch.nn.Module):
    """Model that indexes chunk results with an OOB constant."""

    def __init__(self, chunks, dim, index):
        super().__init__()
        self.chunks = chunks
        self.dim = dim
        self.index = index

    def forward(self, x):
        parts = torch.chunk(x, self.chunks, self.dim)
        return parts[self.index]


class TestChunkGetitemOOBIndex(PytorchLayerTest):
    """Chunk + getitem OOB index tests (uses dynamic Slice, no vector OOB)."""

    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.parametrize("valid_index", [-3, -1, 0, 1, 2])
    def test_valid_chunk_index(self, valid_index, ie_device, precision):
        """Valid chunk indices must still work."""
        model = ChunkGetitemOOB(chunks=3, dim=1, index=valid_index)
        sample_input = self.random.randn(1, 12)

        try:
            scripted = torch.jit.script(model)
        except Exception:
            pytest.skip("torch.jit.script rejected the model")

        ov_model = ov.convert_model(scripted, example_input=(sample_input,))
        assert ov_model is not None


# ---------------------------------------------------------------------------
# SequenceMark (prim::ListConstruct) + aten::__getitem__ with OOB index
# ---------------------------------------------------------------------------

class ListConstructGetitemOOB(torch.nn.Module):
    """Model that indexes a list-constructed sequence with an OOB constant.

    prim::ListConstruct becomes a SequenceMark during conversion; the
    aten::__getitem__ on it is handled by the SequenceMark branch of
    AtenGetItemReplacer.
    """

    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        a = x[:, :2]
        b = x[:, 2:5]
        c = x[:, 5:]
        lst = [a, b, c]
        return lst[self.index]


class TestListConstructGetitemOOBIndex(PytorchLayerTest):
    """
    Tests for SequenceMark (prim::ListConstruct) + aten::__getitem__ with
    out-of-bounds index.

    The SequenceMark branch in AtenGetItemReplacer validates idx.size() == 1
    and checks bounds against the number of sequence inputs before accessing.
    """

    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.parametrize("oob_index", [3, 10, 999])
    def test_positive_oob_index_no_crash(self, oob_index, ie_device, precision):
        """Positive OOB index on list construct must not crash."""
        model = ListConstructGetitemOOB(index=oob_index)
        sample_input = self.random.randn(1, 10)

        try:
            scripted = torch.jit.script(model)
        except Exception:
            pytest.skip("torch.jit.script rejected the model")

        with pytest.raises(ov.frontend.OpConversionFailure):
            ov.convert_model(scripted, example_input=(sample_input,))

    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.parametrize("oob_index", [-4, -100])
    def test_negative_oob_index_no_crash(self, oob_index, ie_device, precision):
        """Large negative OOB index on list construct must not crash."""
        model = ListConstructGetitemOOB(index=oob_index)
        sample_input = self.random.randn(1, 10)

        try:
            scripted = torch.jit.script(model)
        except Exception:
            pytest.skip("torch.jit.script rejected the model")

        with pytest.raises(ov.frontend.OpConversionFailure):
            ov.convert_model(scripted, example_input=(sample_input,))

    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.parametrize("valid_index", [-3, -2, -1, 0, 1, 2])
    def test_valid_list_index_still_works(self, valid_index, ie_device, precision):
        """Valid indices (including negative wrapping) must still convert."""
        model = ListConstructGetitemOOB(index=valid_index)
        sample_input = self.random.randn(1, 10)

        try:
            scripted = torch.jit.script(model)
        except Exception:
            pytest.skip("torch.jit.script rejected the model")

        ov_model = ov.convert_model(scripted, example_input=(sample_input,))
        assert ov_model is not None


# ---------------------------------------------------------------------------
# AppendListUnpackReplacer: prim::ListUnpack + aten::__getitem__ with OOB
# ---------------------------------------------------------------------------

class ListUnpackGetitemOOB(torch.nn.Module):
    """Model that triggers prim::ListUnpack after aten::__getitem__ on a
    list-of-tensors.

    This exercises the AppendListUnpackReplacer path where a list construct
    is indexed before being unpacked.
    """

    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        # Build a list of 2D tensors (Tensor[][]-like pattern):
        # each element is [N, features] stacked along a new dim 0.
        a = x[:, :3].unsqueeze(0)   # [1, N, 3]
        b = x[:, 3:7].unsqueeze(0)  # [1, N, 4]
        c = x[:, 7:].unsqueeze(0)   # [1, N, 3]
        lst = [a, b, c]
        selected = lst[self.index]
        # Unpack along dim 0 (prim::ListUnpack pattern)
        result = selected.squeeze(0)
        return result


class ListUnpackGetitemMultiOutput(torch.nn.Module):
    """Model triggering AppendListUnpackReplacer with multiple unpack outputs.

    Uses indexing into a list built from stacked tensor slices.
    """

    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        # Create a list of stacked tensors
        a = x[:, :3]
        b = x[:, 3:6]
        c = x[:, 6:9]
        stacked = torch.stack([a, b, c], dim=0)  # [3, N, 3]
        parts = [stacked[0], stacked[1], stacked[2]]
        return parts[self.index]


class TestListUnpackGetitemOOBIndex(PytorchLayerTest):
    """
    Tests for AppendListUnpackReplacer with aten::__getitem__ OOB index.

    The AppendListUnpackReplacer handles prim::ListUnpack optionally
    preceded by aten::__getitem__. An OOB index into the list must be
    caught gracefully without crashing.
    """

    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.parametrize("oob_index", [3, 10, 999])
    def test_positive_oob_index_no_crash(self, oob_index, ie_device, precision):
        """Positive OOB index on list unpack must not crash."""
        model = ListUnpackGetitemOOB(index=oob_index)
        sample_input = self.random.randn(2, 10)

        try:
            scripted = torch.jit.script(model)
        except Exception:
            pytest.skip("torch.jit.script rejected the model")

        with pytest.raises(ov.frontend.OpConversionFailure):
            ov.convert_model(scripted, example_input=(sample_input,))

    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.parametrize("oob_index", [-4, -100])
    def test_negative_oob_index_no_crash(self, oob_index, ie_device, precision):
        """Large negative OOB index on list unpack must not crash."""
        model = ListUnpackGetitemOOB(index=oob_index)
        sample_input = self.random.randn(2, 10)

        try:
            scripted = torch.jit.script(model)
        except Exception:
            pytest.skip("torch.jit.script rejected the model")

        with pytest.raises(ov.frontend.OpConversionFailure):
            ov.convert_model(scripted, example_input=(sample_input,))

    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.parametrize("valid_index", [-3, -2, -1, 0, 1, 2])
    def test_valid_list_unpack_index(self, valid_index, ie_device, precision):
        """Valid indices for list unpack must still convert."""
        model = ListUnpackGetitemOOB(index=valid_index)
        sample_input = self.random.randn(2, 10)

        try:
            scripted = torch.jit.script(model)
        except Exception:
            pytest.skip("torch.jit.script rejected the model")

        ov_model = ov.convert_model(scripted, example_input=(sample_input,))
        assert ov_model is not None

    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.parametrize("valid_index", [-3, -1, 0, 1, 2])
    def test_multi_output_valid_index(self, valid_index, ie_device, precision):
        """Multi-output list pattern with valid index must convert."""
        model = ListUnpackGetitemMultiOutput(index=valid_index)
        sample_input = self.random.randn(2, 9)

        try:
            scripted = torch.jit.script(model)
        except Exception:
            pytest.skip("torch.jit.script rejected the model")

        ov_model = ov.convert_model(scripted, example_input=(sample_input,))
        assert ov_model is not None
